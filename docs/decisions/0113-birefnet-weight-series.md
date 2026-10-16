# 0113: BiRefNet 系の重み格納系列 — 段 1（f16）の形・丸めの位置・品質計測の門

- Status: accepted（段 1 = f16。i8 は段 2 で本 ADR を拡張する）
- Date: 2026-09-26
- 関連: ADR [0006](0006-quantization.md)（fake-quant 方法論）/ [0018](0018-f16-weight-execution.md)（f16 格納の
  実行）/ [0027](0027-sbv2-f16-series.md)（f16 系列の先行形・決定 3 の `--verify` 排他）/
  [0029](0029-sbv2-i8-series-and-quant-quality.md)（系列 × 格納 dtype の検出限界と集合一致検査）/
  [0050](0050-irodori-quant-series.md)（irodori 波 1 = 本 ADR の写し元）/ [0075](0075-quant-presentation.md)
  （quant 席の表示欄）/ [0092](0092-distribution-repos-and-sources.md)（BiRefNet の 2 リポ × 2 解像度）。
  判断材料の調査は高速化 / メモリの波の深掘り E1（`.claude/reviews/2026-09-25_perf-recon/deep/E1-birefnet-series.md`・
  git 追跡外）

## Context

BiRefNet 系（BiRefNet_HR と、その fine-tune の Lucida）の配布形は f32 の 1 系列だけで、重みの GPU
常駐は 1024² で 919 MiB・2048² で 1,116 MiB あった。f32 だけなのはカーネルの制約ではない —
conv2d の直接カーネルと implicit GEMM はどちらも `w=f16` / `w=i8` を持ち（ADR 0018 / 0019）、足りない
のは recipe の dtype 軸と品質の裁定だけだった。

調査で分かった事実が 3 つある:

1. **上流 BiRefNet_HR の checkpoint は f16 である**（safetensors ヘッダ F16 687 本 + I64 67 本・F32 0 本）。
   export は `dtype=torch.float32` で読んで f16 の値を f32 へ広げて焼いていた。Lucida は F32 687 本。
2. **⑦（`BatchNorm2d` → per-channel の `x·α + β`）は BN を conv 重みへ畳まない**。α / β は BN の統計から
   f32 で導く派生定数で、mul / add の入力（重みスロットではない = 適格外）なので f32 で格納される。
3. **適格（圧縮格納のまま GPU 常駐できる重み）は 1024² で 89.2%・2048² で 73.5%**。窓マスク 8 本・
   ゼロ pad 定数 16 本・deform_conv2d の重み 20 本（重みスロットを持たない）が f32 のまま残る。

i8 は写しでは足りない（窓 attention の相対位置表と TailConv が `nn.Linear` / `nn.Conv2d` のモジュールでなく
scale が作られない・deform の `regular_conv` は丸まるのに f32 で格納される）ので、利用者の承認
（2026-09-26）は **2 段に割る案**: 段 1 = f16 + 品質計測の器、段 2 = BiRefNet 固有の i8 計画 + i8 系列。

## Decision（段 1）

1. **f16 は別系列**（`outputs/series/<モデル>-<解像度>-f16/` — `birefnet-hr-1024-f16` など）。f32 系列の
   綴りは接尾なしのまま動かさない。同居させると f32 系列の網（系列ごとの tolerance）が圧縮資産へ黙って
   掛かる（ADR 0018 / 0027 と同じ理由）。`--dtype {f32,f16}`（既定 f32）は `--verify` と排他（ADR 0027
   決定 3 と同じ構造理由 — `--verify` の参照はパッチ前の f32 重みの eager）。i8 は CLI が受け付けない。
   〈→ 追記 2026-10-11: 書ける dtype と既定は checkpoint ごと — HR は f16 だけ・既定 f16〉
2. **丸めの位置は `load_model` の直後・`patch.apply` の前**（`birefnet.export.load_wrapper`）。apply の
   後に丸めると ⑦ の α / β まで f16 の値へ動き、格納は f32 のままなので VRAM は減らず数値だけが動く。
   apply の前なら α / β は丸めた統計から f32 で導かれる。丸めは `round_weights_to_f16`（全パラメータと
   f32 バッファ）で、emit の適格判定が重みスロットだけを f16 格納にする。
3. **HR の f16 系列は上流に対して無損失**（丸めが恒等）。golden は f32 系列と**ビット一致**する
   （2026-09-26 に 1024² の 8 ケースで入出力とも突合済み）。Lucida の f16 系列は有損失。
4. **品質計測の器（export 時・torch CPU・GPU 不要）**: 同じプロセスで丸めの**前**に f32 の出力（別に
   読んだ f32 のラッパ・パッチ後のグラフ）を採り、丸めた重みの出力との差を系列ディレクトリの
   `quality.json` に書く。ケースは合成 4 + 実画像 4 の 8 つ（実画像は `--real-images` の有無に依らず
   通す）。指標は logit の max_abs / rel_rms・二値マスクの不一致率・sigmoid 後の α の平均絶対差・
   8 bit α が 1 段以上動いた画素の割合と最大段数・ビット一致の有無。
5. **品質の門（公開の前に掛ける — 落ちたら系列ごと消える）**:
   - checkpoint の浮動小数が全て f16 なら（期待は checkpoint のヘッダから引く — 丸めた結果からは
     導かない）、全ケースが**ビット一致**でなければ落とす。丸めの位置の退行（apply の後で丸める）を
     捕まえる恒真化チェック。比較は値の `==` ではなくビット列（`torch.equal` は `0.0` と `-0.0` を等しい
     とみなす）。
   - 有損失の checkpoint は、α（sigmoid 後）の平均絶対差の最悪が **0.5 LSB（8 bit α の半段 = 0.5 / 255 ≈
     1.96e-3）以上なら落とす**（`birefnet.export.ALPHA_MAE_LIMIT`）。暫定線で、出典は深掘り E1 §6 の kill
     基準（「Lucida f16: MAE_α ≥ 0.5 LSB なら席を出さない」）。2026-09-26 のレビュー（A-2）で記録だけから
     門へ上げた。Lucida f16 の実測（1024² 7.0e-5・2048² 4.8e-5）は 30 倍近い余裕で通る。線は段 2（i8）の
     実測で見直す。加えて差が**全ケースで 0** なら落とす（基準と比較側が同じ重みを見ている = 計測の器が
     恒真に倒れている形）。
   - 圧縮系列は `--real-images` なしでも、丸めた重みの実画像 4 ケースの出力に前景比の順序検査を掛ける
     （`--real-images` のときは golden の sanity が掛けている）。
6. **配布形**: 出力 path を `matte/model.{f32,f16}.krm`・配置の役割名を `matte_{dtype}` に割る。要求表は
   `matte_f32 → f32` / `matte_f16 → f16`、禁止表は f32 席だけ（圧縮の語彙全部 — f16 席の取り違えは要求
   検査が落とす）。`karume dist --pipeline birefnet|lucida` は両系列が揃わないと組めない
   〈→ 追記 2026-10-11: HR は f16 系列だけで組む〉。検査
   （寸法・出所・`pipelineConfig`）は全 dtype の系列に掛け、`pipelineConfig` の一致と、**全 dtype の系列が
   同じ上流 revision を名乗ること**も見る。ライセンス（HR / Lucida とも MIT）と寸法は checkpoint を跨いで
   一致するので、revision を見ないと「HR の f32 系列 + Lucida の f16 系列」が 1 モデルの 2 席として組める。
   〈→ 追記 2026-10-11: revision の一致は checkpoint の固定 revision との突合に置き換えた〉
7. **quant 席は f32 / f16 の 2 つ・既定は f32 のまま**。既定の変更は配布の意味の変更なので、段 2 の品質
   実測と目視（実画像 4 枚以上のマット PNG 対）を経て裁定する。HR の f16 は無損失だが、公開済みの既定を
   黙って動かさない。（→ 追記 2026-09-27: HR は次リリースで f16 席のみ・Lucida は f32 既定 + f16）
   〈→ 追記 2026-10-11: recipe で実施〉
8. **NOTICE とカードの格納の説明は 1 か所から組む**（`birefnet.card.birefnet_storage_lines`）。checkpoint の
   格納 dtype（manifest に無い事実 — `BirefnetCheckpoint.stored_dtype`）と配る席から、HR は「量子化なし
   （f32 席は正確に広げただけ・f16 席は上流の値そのもの）」、Lucida は「f16 席は量子化」と名乗る
   〈→ 追記 2026-10-11: HR の f32 席は無い〉。
   `base_model_relation` は既定 quant の実体が checkpoint の値を丸めているときだけ `quantized`（既定 f32 の
   間はどちらのリポも置かない）。カードの `stored_dtype` とヘッダの一致は pytest が実重みのある機で見る。
   f16 席の格納の説明は「linear / convolution / embedding（相対位置表）の重みが f16・deform の sampling
   weights（`regular_conv`）と bias / norm は f32・BatchNorm の α / β は統計から f32 で導いた派生値」。
   カードの GPU 総確保（1024² 約 1.7 GiB・2048² 約 4.1 GiB〈訂正 2026-10-11: 1 GiB = 1,024 MiB で約 1.6 GiB・
   約 4.0 GiB — 元の値は MiB を 1,000 で割っていた〉）は f32 席の実測で、カードはそう名乗る
   （f16 席の総確保は未測）。〈→ 追記 2026-10-11: 実測は checkpoint ごと — HR は f16 席の実測〉
9. **TS の門**: `e2e_birefnet_test.ts` に f16 の 4 系列（HR の tolerance は f32 系列と同じ値〈golden が
   ビット一致するため〉・Lucida は f32 系列の値を仮置きし実測で導出する）〈→ 追記 2026-10-11: HR の f32
   系列は退役〉と、容器の**圧縮格納 dtype の
   集合**を系列の宣言と突き合わせる GPU 不要の検査（ADR 0029 決定 2 の形 — f32 系列は空・f16 系列は
   `["f16"]`）。2048² の f16 系列も B570 の held 行に載せる（落ちる deform_conv2d の重みは f16 系列でも
   f32 格納）。実画像 e2e（models 側）にも 1024² の f16 2 系列を足す（参照値の行は実 GPU で作る）。

## 実測（2026-09-26・torch CPU の export）

| 系列                 |    所要 |        適格（f16） |      適格外（f32） |          容器 | 品質（8 ケースの最悪）                                                                                                    |
| -------------------- | ------: | -----------------: | -----------------: | ------------: | ------------------------------------------------------------------------------------------------------------------------- |
| birefnet-hr-1024-f16 |   263 s | 205 本 / 410.0 MiB |  422 本 / 99.2 MiB | 534,689,126 B | 全ケースビット一致（門 = bit-exact・golden 8 ケースは f32 系列ともビット一致）                                            |
| lucida-1024-f16      |   260 s | 205 本 / 410.0 MiB |  422 本 / 99.2 MiB | 534,689,126 B | α MAE 7.0e-5・8 bit α は最大 1 段（1.8% の画素）・マスク不一致 8.6e-6・logit max_abs 2.25（\|logit\| 上端 1078 の飽和域） |
| birefnet-hr-2048-f16 | 1,200 s | 205 本 / 410.0 MiB | 422 本 / 296.2 MiB | 741,278,489 B | 全ケースビット一致（門 = bit-exact・golden 4 ケースは f32 系列ともビット一致）                                            |
| lucida-2048-f16      | 1,186 s | 205 本 / 410.0 MiB | 422 本 / 296.2 MiB | 741,278,489 B | α MAE 4.8e-5・8 bit α は最大 1 段（1.2% の画素）・マスク不一致 3.1e-5・logit max_abs 4.62（飽和域）                       |

重みの常駐は 1024² で 919.2 → 509.2 MiB（−44.6%）・2048² で 1,116.2 → 706.2 MiB（−36.7%）。GPU の出力・tolerance・実行時間とロード時間は未測
（実 GPU で `deno task test:models:birefnet`）。〈→ 追記 2026-10-16: 窓マスクとゼロ pad を焼かない形にして、容器と重みの常駐が縮んだ〉

## Consequences

- 1024² の重み常駐は −410 MiB、配布の DL も約 45% 減る。2048² は中間テンソルが支配項なので効きは小さい
  （重みの常駐は −36.7%）。
- 段 2（i8）の予定: BiRefNet 固有の i8 計画関数 1 本が「丸める対象（include）」と「f32 の明示指定」を
  **同じ走査から**返す（`regular_conv` を外す・相対位置表 24 本と TailConv 2 本を f32 指定・
  offset / modulator は初回は f32）。品質計測の器はそのまま使い、閾値をそこで決める。**未知の 3 点**:
  ① conv2d `w=i8` は実モデルでの出荷実績が無い（初使用）② deform の offset / modulator を i8 にしたときの
  品質（サンプリング位置が動く）③ Lucida の飽和 logit（最大 1078）への影響。
- 配布の反映（`karume-birefnet-hr` / `karume-lucida` の再アップロードと `@karume/models` の pin 更新）は
  リリース時（release-runbook）。
- 既存の f32 系列 4 本（`outputs/series/{birefnet-hr,lucida}-{1024,2048}`）は上流 revision を持たない
  （出所の revision を容器へ焼く変更より前の資産）ため、`dist` の出所の門で落ちる。配布に反映するときは
  `--dtype f32` で焼き直す（`outputs/misc/e1-dist/series/` の焼き直し版が使える — golden は既存 f32 系列と
  ビット一致）。〈→ 追記 2026-10-11: HR の f32 系列は配らないので、焼き直すのは Lucida の 2 本だけ〉
- `birefnet-hr-1024-f16` の `quality.json` は値の `==`（`torch.equal`）で比べた版のコードで作った記録で、
  ビット一致の欄の意味が現行コードと違う（ビット一致そのものは golden のバイト同一で確認済み）。
  リリース前の焼き直しで作り直す。

## 追記（2026-09-26 / 2026-09-27）— HR は f16 席のみ・Lucida は f32 既定 + f16

利用者裁定（2026-09-26「悩ましいけど無損失なら f16 でお願いします」→ 2026-09-27「上流に f32 が無いなら f32 の
選択肢は不要」）: **BiRefNet_HR は次リリースで `f16` 席だけを配布する（f32 席は出さない）**。根拠は上流
`ZhengPeng7/BiRefNet_HR` の重みが `model.safetensors` 1 本（444,473,596 B・F16 687 本 + I64 67 本・F32 0 本 —
2026-09-27 に HF の一覧とローカルの safetensors ヘッダで突合）で、f32 の checkpoint が上流に存在しないこと。karume
の f32 系列は f16 の値を f32 へ広げただけ（golden は f16 系列とビット一致）で、情報が増えずに常駐が倍（1024² で
919 vs 509 MiB）になる席だった。f16 格納の読み出しは core WGSL の `unpack2x16float`（ADR 0018）で optional
feature を要求しないので、席を f16 だけにしても動く機種は減らない。
**Lucida は `f32` 既定 + `f16` 任意**（上流が f32 で f16 は有損失 — α の平均絶対差は最大 7.0e-5〈1024²〉/
4.8e-5〈2048²〉・8 bit の α で最大 1 段・二値マスクの不一致は最大 3.1e-5。小さいが 0 ではない）。
配布の意味の変更なので、配布反映（HF 再アップロード + pin 更新）と同じ回に recipe（`dist` の HR を f16 席 1 つで
組める形・既定席・カードの `base_model_relation`・pytest）を変え〈→ 追記 2026-10-11: recipe とテストは先に
変えた・配布反映は次のリリースのまま〉、HR の f32 系列 2 本
（`outputs/series/birefnet-hr-{1024,2048}`）とその golden / sha 参照行・`HELD_SERIES` の f32 行を退役する
（f16 系列の golden が同じ値を持つので検証の被覆は減らない）。決定 6 の「両系列が揃わないと組めない」は Lucida
にだけ残る。段 2（i8）の裁定はそのときの品質実測で行う。

## 追記（2026-10-11）— HR の f32 席の退役を recipe とテストへ入れる

前の追記の裁定（HR は `f16` 席だけ・Lucida は `f32` 既定 + `f16`）を、配布反映より先に recipe とテストへ入れる。
配布反映（`karume-birefnet-hr` / `karume-lucida` の HF 再アップロードと `BIREFNET_SOURCES` の pin 更新）は
次のリリースの回のまま。

1. **席の導出規則**: quant 席は checkpoint ごとに上流 checkpoint の格納の型（`BirefnetCheckpoint.stored_dtype`）から
   導く — **上流の格納より広い席は出さず、既定席は格納と同じ型の席**（`birefnet.distribution` の
   `birefnet_seat_dtypes` / `birefnet_default_quant`）。HR（f16）→ `f16` だけ・既定 `f16`。Lucida（f32）→
   `f32`（既定）/ `f16`。席の表を checkpoint ごとに手で持たず、系列の取得・配置・weights・quant 表・既定席・
   NOTICE・`pipelineConfig` の基準の系列はすべてこの席の集合から引く。HR の配置は f16 系列だけを要求し、
   手元に f32 系列があっても見ない（席に無い dtype の系列を渡された計画は落とす）。Lucida の「両系列が揃わないと
   組めない」（決定 6）は残る。quant 席の表示欄は（席, 上流の格納）の組で引き、HR の f16 席は並ばない f32 席との
   比較を名乗らない（`Upstream precision (f16)`）。カードと NOTICE は上流の格納より広い席の文面を持たず（その
   席を名乗る manifest は描かない）、席が 1 つなら見出しを単数で書く。
2. **export も同じ規則**: `--model-dir` の checkpoint を取得記録の revision（固定値に当たるとき）→ ディレクトリ名
   （系列名と同じ紐づけ）の順で引き、両方が別の checkpoint を指せば落とす（`birefnet.export.model_dir_checkpoint`）。
   同じ関数で書ける dtype と既定を決める。HR の既定は f16、HR に `--dtype f32` を渡すと argparse の error で理由つきで
   拒む。Lucida は既定 f32・f16 も可。帰属表に無い `--model-dir` は語彙の全部・既定 f32（`dist` が組まない）。
   出力先の系列名（`birefnet-hr-1024-f16` など）は変えない。`--verify` は系列を書かないので変えない。
3. **取り違えの検査は固定 revision に置き換える**: `BIREFNET_CHECKPOINTS` の各 checkpoint が上流 revision を
   固定で持ち（HR = `a7a562f6fd16021180f2f4348f4de003a2d3d1e1`・Lucida =
   `6cbedc9722652dc9a3df91dd871f0c4f3334e922` — `inputs/birefnet` の `hf download` の取得記録の値）、`dist` は
   全容器の `provenance.upstreamRevision` をその値と突き合わせる（`assert_upstream_provenance` — Wan と同じ
   形）。決定 6 の「全 dtype の系列が同じ revision を名乗る」検査は消した。理由: HR の席が 1 つになると席どうしの
   一致は何も比べず、「HR のリポに Lucida の系列」を落とせない。固定値で全容器を見れば、席が 1 つでも 2 つでも
   別 checkpoint・別 revision の系列は組めず、席どうしの一致もこれに含まれる。固定値と手元の取得記録の一致は
   pytest が取得記録のある機で見る。
4. **カードの実行資源は checkpoint ごと**（`BirefnetCheckpoint.resources` — 測った席・条件・解像度ごとの数）。
   「実測していない席の数は名乗らない」は、測った席が manifest に無いモデルのカードを描かないことで保つ。
   HR は f16 席の実測（2026-10-11・RTX 3080 Ti）。Lucida の f32 席も同じ日・同じ機・同じ手順で測り直した
   （総確保 1,684 / 4,128 MiB — 前の 2026-09-05 の実測を置き換えた）。総確保は runtime の見積りの
   `peakAccountedBytes`（重み + 中間の領域 + 入力 + 出力の readback）。

   | HR f16 席 |      重み | 中間の領域 |   入力 |   出力 | 総確保（1 GiB = 1,024 MiB） | 最大の binding | 1 枚         |
   | --------- | --------: | ---------: | -----: | -----: | --------------------------: | -------------: | ------------ |
   | 1024²     | 509.2 MiB |  749.2 MiB | 12 MiB |  4 MiB |     1,274 MiB（約 1.2 GiB） |        320 MiB | 1.41〜1.51 s |
   | 2048²     | 706.2 MiB |  2,948 MiB | 48 MiB | 16 MiB |     3,718 MiB（約 3.6 GiB） |      1,280 MiB | 7.25〜7.88 s |

   中間の領域と最大の binding（decoder の `cat_211`）は f32 席と同じ（活性値は f16 席でも f32）。〈→ 追記 2026-10-16: 窓マスクの形を変えた後に測り直した〉
5. **退役したもの**: HR の f32 系列 2 本（`birefnet-hr-{1024,2048}`）を読むテストの行（runtime / models の e2e の
   `SERIES`・`series-graphs.ts`）、HR の f32 の tolerance 定数（実測表と導出は f16 の定数へ移した）、
   `HELD_SERIES` の `birefnet-hr-2048` の行、sha の参照値の `birefnet-hr-1024/*` の 4 ケース。ADR 0125 決定 6 の
   fixture の HR の入力は f16 席の logit から採り直した（ケース名・値・sha256 は同じで、差分は来歴のファイル名と
   席名だけ）。
6. **HR の f16 系列の出力は f32 系列と一致していた**: 実画像 4 ケースのマット logits の sha256 の参照値の行が
   2 環境（RTX 3080 Ti・Arc B570）× 4 ケースとも完全一致した（決定 3 の golden のビット一致に加えて、実 GPU の
   出力も一致）。
7. **Lucida の f32 系列の焼き直しと golden の CPU 依存**: Lucida の f32 系列 2 本を上流 revision つきで焼き直した
   （`dist` の出所の門のため）。krm は part 1 の来歴の欄（+62 B）以外が旧系列とバイト同一。golden（torch CPU の
   参照出力）は全ケースで f32 の丸め誤差ぶんずれた（合成画像で最大 1.8e-4・logit が 1078 に飽和する実画像で
   1.1e-2）。torch は同じ 2.13.0+cpu で、同じ機で 2 回焼くとテンソルはビット同一（違いはヘッダのメタデータの
   並びだけ）なので、**golden は焼いた機の CPU に依存する**（2026-10-06 の開発機の入れ替えで変わった）。焼き直した
   golden に対して既存の tolerance のまま GPU の門が通った（RTX 3080 Ti・`test:models:birefnet` 117 passed）。HR の
   f16 系列は焼き直しても krm と `quality.json` がバイト同一だったので置き換えていない（golden は前の機のまま）。
   手元のミラー 2 本は `birefnet/2` で組み直した（part 31 本は系列の part とバイト同一）。

## 追記（2026-10-16）— 窓マスクを表 + 添字に、ゼロ pad を `expand` にする（xet の断片化の根本原因）

初公開（2026-09-05）でも 0.13.0 の再アップロードでも、2048² の part が HF xet の再構成で細切れになっていた
（[backlog](../backlog.md) の「断片化した part 5 本」・[research 2026-08-09](../research/2026-08-09-xet-fragmentation.md)
の 2026-09-05 追記）。原因はアップロードの手順ではなく、容器のバイト列そのものにあった。

1. **原因 = 焼いた定数の長いゼロ列**: パッチ ④ は shifted-window attention のマスクを `[nW, 144, 144]` の f32 で
   丸ごと焼いていた（8 本・1024² で計 65 MiB・2048² で計 248 MiB — 値は {0, −100} だけで大半が 0）。パッチ ③ の
   ゼロ pad も、エクスポータの定数畳み込みが全ゼロの initializer へ焼いていた（2048² で計 28 MiB）。旧
   `birefnet-hr-2048-f16` の part 00003（253 MiB）は 4 KiB ページの 86% がゼロで、128 KiB 以上のゼロ列が 94 本
   （計 217 MiB・最長 3.3 MiB）あった。xet の content-defined chunking は長いゼロ列の中を同じ全ゼロのチャンクで
   刻み、再構成はその同じ範囲を何度も指す term に割れる。
2. **確かめ方**: HF に触れず、同じ hf_xet（1.6.0）でローカルの CAS へ上げる模擬で比べた。旧 part 00003 は
   265 term（うち 245 本が同じ範囲の繰り返し参照）、新しい part 00003 は 4 term（繰り返し 0）。HF 上の旧 part 00003
   の再構成も、term が多いのに取得レンジは少ない繰り返し参照の形だった（HR 2048² f16 = 109 term のうち 93 本・
   Lucida 2048² f32 = 432 term のうち 342 本が繰り返し参照・取得レンジは 9 / 23 本）。繰り返し参照はサーバ側の
   dedup を使わない模擬でも出るので、dedup の当たり外れではなくバイト列に由来する。term の数の絶対値は模擬と HF で
   一致しない（旧 1024² の part 00003 は模擬 132 term・HF 10 term — HF 側のまとめ方の違いと見るが未確認）。
   調査記録は `outputs/diag/birefnet-xet-2026-10-16/`（git 追跡外）。
3. **直し方**（`birefnet.patch` の ③ と ④）: マスクは窓の種類（内部 / 右端の列 / 下端の行 / 右下の角の高々 4 種）
   の表 `[K ≤ 4, 144·144]` と窓ごとの添字（i32 の定数）だけを焼き、グラフの中で `embedding` → `view` で組む。
   組み立て直したマスクが元のマスクとビット一致しなければ export を落とす。ゼロ pad は 1 要素のゼロを `expand` で
   広げる（`expand` は定数畳み込みの frontier なので、焼かれるのは 1 要素だけ）。マスクの表は embedding の表なので、
   f16 席では f16 で格納される（値 {0, −100} は f16 で正確 — f16 の格納は相対位置表 205 本に表 8 本が加わった）。
4. **容器の大きさ**（GPU の出力は変わらない — `test:models:birefnet` は sha256 の参照値の行を変えずに緑）:

   | 系列                                   |              前 |            後 |
   | -------------------------------------- | --------------: | ------------: |
   | birefnet-hr-1024-f16 / lucida-1024-f16 |   534,689,126 B | 452,927,004 B |
   | birefnet-hr-2048-f16 / lucida-2048-f16 |   741,278,489 B | 452,936,766 B |
   | lucida-1024                            |   964,628,992 B | 884,193,864 B |
   | lucida-2048                            | 1,171,218,283 B | 884,203,626 B |

5. **1024² と 2048² で part 00003 以降がバイト同一になった**: 重み本体（マスクの表も含む — 窓の 4 種の値は解像度に
   依らない）が同じになり、違うのは part 00001（グラフと記述子 — 形）と part 00002（グラフの定数 — 窓ごとの添字
   など・約 0.1 MiB）だけ。ミラー（`karume.json` の part の sha256 で確かめた）は part 00003 以降を `shared/` に
   1 本ずつ置く: `karume-birefnet-hr` は 9 → 6 ファイル・1,216.9 → 432.8 MiB、`karume-lucida` は 22 → 14 ファイル・
   3,253.8 → 1,276.8 MiB。
6. **実行資源**: マスクは常駐の重みから実行時の中間へ移った（最大は 2048² の `[1849, 144, 144]` f32 ≈ 153 MB）。
   1024² の中間の領域は 1.5 MiB 増え、2048² は生存ピークの内側に収まって変わらない。カードの値を同じ手順
   （runtime の見積りの `peakAccountedBytes`・ウォームアップ 1 回の後の 5 回）で測り直した（RTX 3080 Ti）:

   | 席               |      重み | 中間の領域 | 入力 + 出力 | 総確保（1 GiB = 1,024 MiB） | 最大の binding | 1 枚         |
   | ---------------- | --------: | ---------: | ----------: | --------------------------: | -------------: | ------------ |
   | HR f16 1024²     | 431.2 MiB |  750.7 MiB |      16 MiB |     1,198 MiB（約 1.2 GiB） |        320 MiB | 1.42〜1.51 s |
   | HR f16 2048²     | 431.2 MiB |  2,948 MiB |      64 MiB |     3,443 MiB（約 3.4 GiB） |      1,280 MiB | 6.95〜7.20 s |
   | Lucida f32 1024² | 842.5 MiB |  750.7 MiB |      16 MiB |     1,609 MiB（約 1.6 GiB） |        320 MiB | 1.41〜1.49 s |
   | Lucida f32 2048² | 842.5 MiB |  2,948 MiB |      64 MiB |     3,854 MiB（約 3.8 GiB） |      1,280 MiB | 6.70〜6.93 s |

   Lucida の run は GPU を冷ましてから 2 系列だけを測った値。HR の 2 系列の直後に続けて測ると、1024² が
   1.69〜2.01 s と回ごとに遅れた（2026-10-11 も Lucida は単独で測っている）。生データは
   `outputs/misc/birefnet-resources-2026-10-16/`（git 追跡外）。
