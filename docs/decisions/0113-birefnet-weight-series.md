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
   検査が落とす）。`karume dist --pipeline birefnet|lucida` は両系列が揃わないと組めない。検査
   （寸法・出所・`pipelineConfig`）は全 dtype の系列に掛け、`pipelineConfig` の一致と、**全 dtype の系列が
   同じ上流 revision を名乗ること**も見る。ライセンス（HR / Lucida とも MIT）と寸法は checkpoint を跨いで
   一致するので、revision を見ないと「HR の f32 系列 + Lucida の f16 系列」が 1 モデルの 2 席として組める。
7. **quant 席は f32 / f16 の 2 つ・既定は f32 のまま**。既定の変更は配布の意味の変更なので、段 2 の品質
   実測と目視（実画像 4 枚以上のマット PNG 対）を経て裁定する。HR の f16 は無損失だが、公開済みの既定を
   黙って動かさない。（→ 追記 2026-09-27: HR は次リリースで f16 席のみ・Lucida は f32 既定 + f16）
8. **NOTICE とカードの格納の説明は 1 か所から組む**（`birefnet.card.birefnet_storage_lines`）。checkpoint の
   格納 dtype（manifest に無い事実 — `BirefnetCheckpoint.stored_dtype`）と配る席から、HR は「量子化なし
   （f32 席は正確に広げただけ・f16 席は上流の値そのもの）」、Lucida は「f16 席は量子化」と名乗る。
   `base_model_relation` は既定 quant の実体が checkpoint の値を丸めているときだけ `quantized`（既定 f32 の
   間はどちらのリポも置かない）。カードの `stored_dtype` とヘッダの一致は pytest が実重みのある機で見る。
   f16 席の格納の説明は「linear / convolution / embedding（相対位置表）の重みが f16・deform の sampling
   weights（`regular_conv`）と bias / norm は f32・BatchNorm の α / β は統計から f32 で導いた派生値」。
   カードの GPU 総確保（1024² 約 1.7 GiB・2048² 約 4.1 GiB）は f32 席の実測で、カードはそう名乗る
   （f16 席の総確保は未測）。
9. **TS の門**: `e2e_birefnet_test.ts` に f16 の 4 系列（HR の tolerance は f32 系列と同じ値〈golden が
   ビット一致するため〉・Lucida は f32 系列の値を仮置きし実測で導出する）と、容器の**圧縮格納 dtype の
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
（実 GPU で `deno task test:models:birefnet`）。

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
  ビット一致）。
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
組める形・既定席・カードの `base_model_relation`・pytest）を変え、HR の f32 系列 2 本
（`outputs/series/birefnet-hr-{1024,2048}`）とその golden / sha 参照行・`HELD_SERIES` の f32 行を退役する
（f16 系列の golden が同じ値を持つので検証の被覆は減らない）。決定 6 の「両系列が揃わないと組めない」は Lucida
にだけ残る。段 2（i8）の裁定はそのときの品質実測で行う。
