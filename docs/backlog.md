# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — 0.12.0 リリース後（2026-09-06）

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
  段 3 の作るもののうち HF の全 pin の移行（残り 9 リポの再アップロード）は release の波へ移した（release 節の
  「HF 配布リポの `karume/5` 再アップロード」項が正本）。段 4〜6 の作るものと検収は ADR 0108 の段階分解の表が正本。

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
  sequential 席で投機 / 非投機の 200 token × 3 ケース厳密一致・受理 1.51 / 2.01 / 2.14 token/cycle）。**段 4-A ✅ 2026-09-09**（`tools/mtp-bench` `5b73fc4`・demo `--speculative` `118689a`・[research 2026-09-09](research/2026-09-09-mtp-stage4.md): 抽出 **1.81×** / 要約 1.41× / 対話 1.18× / 自由文 0.96×〈decode 相・k=3〉・採算 A / A\* で A\* = 1.72〜1.88・温度 1.0 でも受理率は greedy と同じ・token 列一致・長文脈の verify 超過 +5.7 ms は attention ①・draft 壁 15 ms の ≈9.5 ms は Deno の round trip・抽出は k=3 飽和。**M2**〈ユーザー実走〉: 抽出 1.27× / 要約 1.03× / 対話 0.93× / 自由文 0.76×・A\* = 2.17〜2.6・verify が decode の 1.75〜2.2 倍〈短文脈で既に +39 ms = M=4 linear 側〉・prefill 4.8K 54 s〈別項〉）。**4-B ④ ✅ 2026-09-09**（`f845f55` ゲート v2〈16 cycle ブロック × 2 連続で抜ける・8 cycle バーストで戻る〉・`63c6db3` 受理列挙の停止・`cca5a36`/`7f02818` mtp-bench 3 値・ADR 0096 決定 8・[research §6.1](research/2026-09-09-mtp-stage4.md): RTX で auto ≈ always〈勝つ 3 条件 +1〜4%・自由文 0.965× vs always 0.948×〉・M2 の検収はユーザー実走待ち）。**残り = 4-B（承認 2026-09-09・反証後の順序: ④ 自己採算ゲート〈壁と受理の EWMA・非対称ヒステリシス 0.97 / 1.01・plain 側は幾何バックオフ 8→256・W1 は cycle 2 で先に採る・既定 on = 既定席では同一 seed でも稀に出力が変わりうると limitations に明記・`speculative: boolean | "always"`・前段として受理列挙を停止 token で止める仕様変更を別コミット〉→ ⑧ M2 の小 M linear〈**測定済み・否定** 2026-09-09: 静的ノブ `linearGemvRowsThreadTarget`（`87e702b`）で r1 / r2 / r4 を M2 で比べると既定 r1 が最良（verify 90.7 → 103.6 ms）・「重み 4 回読み」仮説は外れ・backend 別既定は不要・K-22 は機序の宿題として残す〉→ ⑤ は縮小〈③′ は既に M=4 で効いており M=1 限定は ①QK だけ・①′ の WGSL は既に M 一般なので適用条件 1 行 + テスト 3 本 + kernel doc の MUST・効き代は M2 で最大 −4 ms/cycle・非負なら残す〉→ ③ argmax → ⑥ k=7 + 3 値ゲート・⑦ 不採用）**。**⑤ ✅ / ③ ✅ 2026-09-09**（`06f82df` ①′ を M ≤ 8 へ〈RTX 中立・既定席で verify 行 0 が u32 一致〉/ `e370562` 2 相 argmax〈drafter の argmax 2.03 → 0.08 ms/run・ビット同一〉— [research §6.2 / §6.3](research/2026-09-09-mtp-stage4.md)）。**M2 検収 ✅ 2026-09-09**（抽出 auto 1.379× / 対話 0.966× / 自由文 0.874× — ゲートは負けを半分に・0.99× には届かず → 残件 = ゲート v3〈強い信号の早抜け・バースト初期間隔 16・bench の sequence 使い回し〉）。**残り = ⑥（裁定待ち: parked か 2 drafter + 3 値ゲート）・ゲート v3・⑩・4-C docs**。**⑨ ✅ 2026-09-09**（`0c8bacd` per-cycle トレース = `onRun` の壁・確定数・ゲート状態 + mtp-bench の局面別バケット・[research §6.4](research/2026-09-09-mtp-stage4.md): RTX では勝つ側は抜けず W1 プローブ 1〜3% のみ・cold は 1%・**ゲートが落とす plain step は長文脈で真の decode より +17〜30% 高い〈verify 形・物理行は 1 で原因は未帰属〉→ 新規 ⑩ = 原因の帰属〈gpu-timing をモード別に割る〉と対策**・M2 の trace〈[research §6.4.1](research/2026-09-09-mtp-stage4.md)〉: 自由文の損の 62% は探索バースト・36% は抜けるまで・対話は 6% 負けを 63 cycle 続ける費用・v3 = 早抜け + 間隔 16 で ≈ 0.94×・warm start で ≈ 0.97×）。**ゲート v3 ✅ 2026-09-10**（`29e74b0` / `15b8111`・[research §6.5](research/2026-09-09-mtp-stage4.md)・ADR 0096 追記: RTX cold は勝つ側で誤退出 0・W1 半減・M2 の v3（[research §6.5.1](research/2026-09-09-mtp-stage4.md)）: cold 自由文 0.934× / 対話 0.959× / 抽出 1.361×・**warm 自由文 0.978× / 対話 0.972×**（残りは 1 ターン 1 回のバースト → exploreMax 512 級で 0.99× 圏）。warm の口は `746ad77` でターンごとに違う発話に直し写しは消えた（research §6.5 表 11′）。**新知見: warm の対話（勝つ課題）で 3 / 7 ターンが抜けて auto / always 0.96 → A/B 済み（`063c874`・research §6.5 表 11″）: 帰属は v3 の 1 ブロック早抜け → **v3.1 ✅ 2026-09-10**（`a28e56f`・`earlyLeave` 既定 off / `burstAbort` 0.15・RTX warm 対話 auto = always・[research §6.5.2](research/2026-09-09-mtp-stage4.md)）。exploreMax 256 → 512（`e365d9a`）。M2 warm v3.1（exploreMax 256 ビルド）= 自由文 0.990× / 対話 0.981× で、512 の検収は M2 warm の再計測待ち。⑩ は host / readback 側と判明・帰属は run 内の相の壁が要る）。旧記述（⑤ の −6 ms / u32 同一は反証で撤回）= ③ argmax の 2 相化（drafter −1.5 ms/cycle・ビット同一が構造保証）④ 投機の on / off ゲート（k′ ∈ {0, k} を受理の移動平均で・`accepted` の切り詰め前加算の誤りも修正）⑤ ①′ / ③′ の行タイル化（verify M ≤ 8 を decode と同じレーン割り → attention −6 ms + 既定席で u32 同一・着手条件 = 4-A で attention ≥ 5 ms/cycle）⑥ k=7 drafter の再 export + 動的 k（chat 形式の E[a] で予測倍率が k=3 の 1.2× 以上のとき）⑦ GPU argmax / topk 出口（readback + ホスト受理が cycle の 10% 以上のとき）→ 4-C docs（ADR 0096 の段表 段 4 行を現状へ消し込む — 採算表と「fence 1 本化は不成立」は ADR 追記と research §5 に記録済み）。裁定: ブラウザ実測はリリース後・M2 のミラーはユーザーが scp。事前プローブ（RTX・P≈4.8K・k=3・greedy）: decode 29.0 ms/run〈GPU 25.2〉・verify 37.0〈GPU 35.4 = decode + attention ①/③′ +6.0 + linear +2.2 + lm_head +0.8〉・draft 15.7〈GPU 6.5・argmax 1.8〉→ cycle 52.7 ms / 2.15 token = 24.5 ms/token = **1.18×**（decode 相・ブラウザ相当 ≈1.3×）。旧記述: 実測 Deno / Metal・温度 1.0 での受理率・投機を張る文脈長の閾値・動的 k・GPU argmax・
  fence 削減・①′ の位置不変化）。
- **`planBackingBudgetBytes` を共通の options へ**（起票 2026-09-07 — ADR 0095 帰結）: gemma4 以外は manifest の `session` から
  Session options を組むため予算を変える口が無い（既定 256 MiB が効く）。`onRetry` を `FromPretrainedHubOptions` へ 1 本化した形に
  倣って載せる。併せて estimate / session-build に二重にある予算の値域検査を 1 関数へ寄せる。
- **GEMV 行ブロックの残件（起票 2026-09-07・K-21 の帰結）**: ①並列度の目標 16384 は RTX 3080 Ti の飽和点（limitations）—
  M2 の再掃引は済（既定 16384 が最良）で、残るのは内蔵 GPU の掃引。差し替え口は静的な Session オプション
  `linearGemvRowsThreadTarget` ②ブラウザ（Chrome / Tint）のシェーダ解析費は未測（Deno / naga で初回ターン
  +85 ms）③Metal の u32 門は行ブロック 13 形も未実測（known-issues）。
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
- **既公開 3 リポの `LICENSE.md` / `NOTICE.md` 同梱是正**（起票 2026-09-04 — ADR
  [0092](decisions/0092-distribution-repos-and-sources.md) 決定 7）: `karume-irodori-v4-small` /
  `karume-irodori-v4.1-small`（MIT = 全文 + 著作権行）と `karume-sbv2-jvnv`（CC BY-SA）は
  法的テキストの同梱が漏れている（`verify_dist` の `LEGAL_PATHS` 席）。**次にこの 3 リポを
  上げ直す回に同乗**させる（2026-09-04 ユーザー裁定 — 是正単独の再アップはしない）。
  その回は release 節の `karume/5` 再アップロードで、3 リポとも同乗させる。
  未公開の vowel-detector は同梱済み（2026-09-05 — `PIPELINE.root_files` に MIT 全文 +
  著作権行）なので、初回公開時に漏れることはない。
  同じ上げ直し波に**カード / NOTICE の常時分割の文面是正**も乗せる（ADR
  [0071](decisions/0071-manifest-v3-shards.md) 末尾の未履行記録）: `card.py` の overview 3 本
  （sbv2 / irodori / vowel_detector）と `distribution.py` の
  NOTICE 改変列挙 3 本（siglip2 / depth_anything / gemma4）が単一ファイルの綴りのまま、
  anima の `CONTAINER_MODIFICATION` は「収まらないときだけ分割」の条件つき文面のまま。
- **公開済み `karume-depth-anything-v2` のカード / NOTICE.md 再発行**（起票 2026-09-05）:
  depth-anything の `CONVT_MAXDIFF` は実重み `--verify` の再実測で 1.4e-06 → 6.1e-06 へ確定した
  （合成 4 ケースの最大 — 旧値は 1 ケースぶん。`verify_patches` に上限比較の門も入った）ので、
  公開済みカードと NOTICE.md が名乗る 1.4e-06 は古い。次にこのリポを上げ直す回に同乗させる
  （是正単独の再アップはしない — 上の 2 リポと同じ扱い）。
- **公開済み `karume-birefnet-hr` / `karume-lucida` のカード再発行**（起票 2026-09-06）: `card.py` の
  資源表の「最大 binding」を 256 / 878 MiB から cat_211 込みの 320 / 1,280 MiB へ訂正した（`d116d7e`）ので、
  公開済みカードの数値は古い。次にこの 2 リポを上げ直す回に同乗させる（是正単独の再アップはしない —
  上と同じ扱い）。
- **テスト被覆の残（起票 2026-09-05）**: `SubmitScheduler` の `#encodeTimedChunk` 内の copy 分岐
  （`packages/runtime/src/gpu/submit.ts`）は依然として未検証。
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
- **exporter core の `karume/__init__` が torch を eager import する**: `karume.dist` / `karume.modelcard`
  だけを使う配布・カード層（recipes の dist ドライバ）でも `import dist` で torch が丸ごと読まれる
  （2026-09-04 実測 — recipes 側は torch 非依存の `measurements.py` へ寄せ済み）。`__init__` の
  re-export を遅延化するか、`karume.dist` / `karume.modelcard` を本体から独立に import できる形にする
  （PyPI `karume` の公開面の設計判断 — ADR 0065 の境界）。
- **examples/ の README 整備**: 残るのは anima / irodori / sbv2 / vowel-detector の 4 ファミリ
  （リポ直下 / models / exporter と同じく英語 — CLAUDE.md）。

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

- **HTTP Range 取得（ADR 0108 段 6）の前倒し候補（起票 2026-09-24・判断はリリース後）**: ADR
  [0109](decisions/0109-manifest-v5-container.md) 決定 7 の前倒し条件は「段 2 の RAM ピーク harness で cold の
  ピークが『part 長 + 重ね合わせ』を超える」こと。段 3e で保持の重複は消えたが、scan 型（Deno の HF 経由）の
  cold はまだ part 長 + 最大 block を超える。残りは hub の保持枠 1 本と GC を待つ part で、原因は part 単位の
  全量読み（取得の粒度）に移った。条件は形式上成り立ったまま（[研究記録](research/2026-09-24-part-length-ram-peak.md)
  の 7・ADR 0108 追記 5）。当たるのは Deno の HF 経由だけ。parked の「hub Range 並列 + prefetch」（断片化対策）とは
  動機が別。
- **gemma4 の run 時間の伸びの帰属（起票 2026-09-24）**: 段 3e の M2 で gemma4 の run が +0.14〜+0.55 s 伸びた。
  run の窓には重みの供給が入らないので、供給経路の遅れではない読み。原因は未切り分け（研究記録の 4 と
  「残った問い」）。
- **縮図に外部の正解を戻す（起票 2026-09-24 — ADR 0108 追記 4）**: 3 codec 混在 × piece 分割の合成モデル
  （`packages/runtime/tests/gpu_memory_container_test.ts`）は、合流層と構築経路を共有する 2 経路（krm と
  メモリ内容器）の一致しか見ていない。戻し方は 2 案。CPU 参照の連鎖（`decodeI4` / `decodeI8` +
  `applyReferenceOp`）で期待値を立てるか、縮図の出力に環境別の sha256 参照行（ADR 0106）を足す。
- **PLE のメモリ内容器のフェンス本数（起票 2026-09-24）**: GPU 常駐席の PLE は piece 1 本 = part 1 本で
  メモリ内容器へ渡すので、Session 構築のフェンスが piece の本数ぶん立つ
  （`packages/runtime/src/format/container/memory.ts` の part 割り）。**推測**の見積りは、E2B の values 約 72 block
  × フェンス 13.0 ms（ADR 0108 決定 9 の Arc B570 実測）で約 0.94 s。段 3e 後はホスト RAM が part 割りに依らない
  ので、piece を束ねて増えるのは staging だけ。未実測。
- **`pack_int2` / `unpack_int2` の置き場（起票 2026-09-24）**: `tools/exporter/src/karume/emit.py` の 2 関数は src に
  呼び手が無く、テスト（`test_i2_storage.py`）だけが使う。i2 のバイト順の正本として src に残すか、テスト helper
  へ移すかを決める。判断には ADR 0097 の意図（exporter が自前で i2 を詰める日が来るか）が要る。
- **`parseSafetensors` の 2 引数形（`byteLength`）の置き場（起票 2026-09-24）**: 公開面
  （`packages/runtime/mod.ts`）にあるが、器の使い回しが段 3d で退役してから本番の呼び手は 0 件で、
  残るのは `packages/runtime/tests/format_safetensors_test.ts` の 2 引数形のケース（doc とテスト名は
  「最大 shard 長の器を使い回す」前提のまま）。公開面から外す（Breaking）か、資産の読み手の口として
  残して doc を現行にするかを決める。
- **i4 の group scale の形の式が 2 箇所にある（起票 2026-09-24）**: 検査側（合流層
  `packages/runtime/src/format/container/bind.ts` の scale block 長と group の刻み）と展開側
  （`packages/runtime/src/format/i4.ts` の `groupScaleShape` → `decodeI4`）が同じ形を別々の式で求める。
  段 3d までは検査側が `format/container.ts` で `groupScaleShape` を共有していた。1 本に戻すか、
  両者の一致を fixture で固定するかを決める。
- **container-v1 §6.2 の codec 台帳と実装のずれ（起票 2026-09-24）**: 仕様の台帳エントリは `decodeCpu` /
  `executableOps` / `wgsl` を持つが、実装の `CodecEntry`（`packages/runtime/src/format/container/codecs.ts`）は
  `layout` / `packing` / `scale` / `grouping` / `zeroPoint` だけで、圧縮のまま常駐できる op の判定は今も別々の
  述語（`packages/runtime/src/runtime/plan.ts`）。仕様を実装へ寄せるか、実装を台帳へ畳むかを決める。
- **実重み golden 11 本の結果記録の包み（起票 2026-09-22）**: 各 e2e に同型の try / catch / record が
  並ぶので、helpers 側に「ケース 1 件を記録付きで回す」薄い包みを置いて重複を消す（size S）。

- **コード品質管理の波の残置（起票 2026-09-22 — 出典は
  [退避した消化済み節](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)〈0.12.0 リリース後〉の同波）**:
  - `Gemma4PipelineOptions` を引数に取る 4 本（`assertSpeculative` / `resolveGemma4PleResidency` /
    `buildGemma4Program` / `speculativeSetup`）の置き場（`pipeline-options.ts` の新設は 2026-09-21 裁定で後回し）。
  - sbv2 の `staticInputDim` は方針が逆向きで共通層の対象外のまま。
  - Unicode 区間表の検査は文言が違い各 family に残っている（二分探索だけ `text/code-ranges.ts` へ寄せた）。
  - `models/src/session/with-session.ts` の単体テストが無い。
  - `reference/ops.ts` と `ops/shapes.ts` の分割（info — 可変状態も循環も無く実害ゼロ）。

- **decode 速度の残り（2026-09-20 に now から移動）**: H-27 段 ②（先行投入・ADR 0066 の opt-in 例外・greedy 限定・期待 Deno −5 / Chrome −2.2 ms）、
  小物 K-48 段 1（rms_norm→SRQ 融合 70 本・0.19 ms）/ K-49（slice 別名化）/ K-50（k+v 連結 GEMV）、K-46（int8 KV — メモリ項目）。
  復活条件: decode 速度を再び主題にするとき（H-27 の前提 = H-28 の GPU 常駐席は済・prefill のアリーナ経路のプール化を先に）。
  候補の採否と kill 基準は [perf-ledger](perf-ledger.md)、帰属は [research 2026-09-19](research/2026-09-19-qat-speed-recon.md)。

- **層内の大融合を塞ぐ 3 契約の裁定（起票 2026-09-19）**: WebML との dispatch 差（1,132 → 約 316 本）のうち約 480 本は
  `windowTouchesState` MUST（ADR 0067）・FusedStep 単一出力 MUST（ADR 0068 決定 1）・atomic last-arriver merge の可搬性判定が同時に塞ぐ。
  個別候補（attention→SRQ・残差 add→SRQ・attention 1 dispatch・Q/K/V 3 出力）は全て反証で閉じたので、契約を緩めるかの設計裁定が先
  （[research](research/2026-09-19-qat-speed-recon.md) §9.1）。裁定前に速度候補として起票しない。
- **ブラウザ動画生成の基盤**（調査 2026-09-10）: Wan2.1-T2V-1.3B を小さな DiT 単体 → 実 token 長の
  attention / FFN → causal Conv3d VAE → scheduler と段寿命の順に検収する案。runtime 語彙と数値契約の判断が先。
  H3 は公開重みの規模・未公開の後段・ライセンス条件から構造調査に留める。
  [構成と容量試算](research/2026-09-10-codex-mtp-optimization.md#動画生成の事前調査-wan-と-minimax-h3)。
- **カードの Usage repo 導出の硬化（起票 2026-08-25）**: `karume.dist` はカードの Usage 例の
  repo 名を**出力ディレクトリ名**から導出するため、越境参照のステージング焼き（`--out` が
  別名）で誤った repo 名がカードに載る（実害 = turbo カードに `-release` 付き誤名が公開されて
  いた — ADR [0078](decisions/0078-anima-sampler-selection.md) Consequences・runbook §0 に
  運用注意を追記済み）。恒久策 = `Pipeline.repo_name` 系の正本から導出し `--out` 名へ依存
  しない形。
- **examples/anima に `--sampler` ノブ（起票 2026-08-25）**: request 側 `sampler` 席
  （ADR 0078）を CLI デモから振れるようにする小改修。
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
- **w8a8 鏡像門の設置**: `e2e_deberta_w8a8_test.ts`（ADR
  [0026](decisions/0026-w8a8-deberta-deployment.md) 決定 3 — `e2e_deberta_test.ts` は移植済み・
  鏡像側だけ未設置。2026-08-16 裁定で起票）。
- **Anima ホスト糊 parity の常設門化**: `sigmaSchedule` / `cfgEulerStep` / `denormalizeLatents` /
  `padSequence` の「fixture と全 4 実装 bit 同一」は recipe README に実測記録として残るだけで、
  `outputs/series/anima-pipeline*` を読む常設テストは Deno / pytest のどちらにも存在しない
  （2026-08-16 判明 — fixture 4 変種は再エミット済みで前提は解消済み）。
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
- **`AssetProgress.path` が越境参照を識別できない（起票 2026-08-25・優先度低）**: 進捗イベントの
  `path` は文字列 1 本で、越境コンポーネント参照（ADR
  [0038](decisions/0038-manifest-v1.md) §7 追記）が入った以上**別リポの同名 path と区別が
  付かない**（取得層の同一性キーは `fileRefKey` へ移ったが、公開イベント側は `path` のまま）。
  消費側がファイル別の進捗を path でキーにすると 2 本が混ざる。埋め方は `repo` / `revision` を
  イベントへ足すか `fileRefKey` を出すか — 公開面の追加なので breaking 波に乗せる。
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

- **HF 配布リポの `karume/5` 再アップロード（残り 9 リポ・起票 2026-09-24 — ADR 0108 決定 18・ADR 0109 決定 9）**:
  pin のある 10 リポのうち `karume/5` を指すのは `irodori-v4.1-small` だけ。残りの `anima` / `anima-extra` /
  `birefnet-hr` / `lucida` / `depth-anything-v2` / `gemma4` / `irodori-v4-small` / `sbv2-jvnv` / `siglip2` の 9 本は
  `karume/4` の revision を指し、HEAD の hub では読めない（旧版パッケージからだけ動く）。`gemma4-qat` は未公開で
  pin が無い。手順は [release-runbook](release-runbook.md) の §0〜§3（bump → `karume dist` で焼き直し →
  旧 `*.safetensors` の削除つきアップロード → 削除後の main で pin）。
  - `irodori-v4.1-small` も焼き直して上げ直す。今の pin はカードが `karume/4` と safetensors 方言を名乗り、
    `LICENSE.md` / `NOTICE.md` も無い（移行 CLI の出力ミラーをそのまま上げたため）。移行 CLI のミラーは
    どのリポも上げない。
  - `anima` を先に上げて main の SHA を確定し、`anima-extra` をその SHA の越境参照で焼き直す（runbook §0）。
    ローカルミラーの extra は旧 `karume/4` の anima revision を指している。上げた後に now 残件の
    「anima-extra 越境の実資産門の復活」を行う。
  - anima の系列 14 本を再 export する（上流 checkpoint から CPU で）。公式 4 変種（turbo-v1.1 / aesthetic-v1.1 /
    turbo-v1.0 / aesthetic-v1.0）と copycat の系列が `outputs/series/` に無く、`karume dist` が組めない。
  - sbv2 の front f16 / i8 はミラーが旧世代の export を移行したもので、系列から焼き直すと initializer 名が
    変わる（値と出力は同じ）。
  - 同乗: now 残件の法的テキスト同梱・カード / NOTICE の文面是正・depth-anything と birefnet・lucida の
    カード再発行。
- **vowel-detector の初回公開の前提（起票 2026-09-24）**: recipe は上流の `feature_config.json` を
  `inputs/vowel-detector/` 直下から読む。この開発機は上流リポを丸ごと置いた形なので、組み立ての前に
  `cp inputs/vowel-detector/assets/feature_config.json inputs/vowel-detector/` を 1 回打つ。
- **`tools/llm-baseline` の lint の門（起票 2026-09-24）**: pyproject も CI ジョブも無く、README の手動の
  `uvx ruff check` / `ruff format --check` だけが頼り（2026-09-24 時点で ruff check は赤）。exporter /
  export-recipes と同じ設定で CI に載せるか、export-recipes の workspace メンバーへ寄せるかを決める。
- 実資産 CI gate（GitHub CI はローカル資産を踏まない問題）。**門番は消化済み**
  （`packages/runtime/tests/assets_gate_test.ts` + CI env `KARUME_ALLOW_NO_ASSETS=1` —
  2026-09-05）。残るのは golden の fixture 昇格 / release gate での資産取得の判断
- リポ直下 README の書き上げ・JSR npm 互換層の sideEffects 検証。**0.8.0 の範囲外
  （2026-09-03 裁定 — Status 行だけスタブと名乗る形へ差し替え済み）。復活条件 = 1.0 または
  対外アナウンス時**で、着手にはバンドルサイズの再実測が前提（2026-08-16 の gzip 実測は
  gemma4 生成 API・GEMV 族・tokenizer の追加で失効している）
- ライセンス interview（export-recipes の family 別 provenance を upstream revision 単位で
  人間確認 — 再編の release gate。**公開 4 リポぶんは波 K-4 の人間ゲートで先行実施**）。
  **公開済みは 7 家族 10 エントリで、重み行（Revision used / Weights license）が未記録なのは
  irodori / sbv2 / siglip2 の 3 家族**（anima / depth_anything / gemma4 / birefnet・lucida は記入済み）。
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
- GPU timestamp 推定源化・全面 f16（案γ）・SBV2 NFC チップ・f32 anima 系列再生成
- by-design 制約群（rank≤4 / OOB NaN / 非有限入力 / 0 要素次元 / cancellation 粒度ほか）は
  limitations.md が正本 — 実需が出た項目だけここへ昇格させる
