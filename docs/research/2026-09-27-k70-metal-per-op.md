> 2026-09-27 時点の実測スナップショット。Apple M2（Chrome 153 / Dawn / metal-3）と Intel Arc B570（BMG G21・Deno 2.9.6 /
> wgpu / Vulkan）の op 別 GPU 時間の突き合わせ。JSON は `outputs/bench-browser/` と
> `outputs/bench/karume/2026-09-27_metal-recon/`（どちらも git 追跡外）。公称性能は各社の公開値で、この機で測った値ではない。

# K-70: Apple / Metal で anima が遅い理由 — 大タイルの GEMM 幾何と i8a8 の Metal 展開

perf-ledger K-70（Apple / Metal〈Chrome〉での anima の遅さの帰属）の M2 実測。確認ページ
`tools/anima-residency/browser` に足した op 別 GPU 時間（`lastRunTiming`）で、M2 の 8 行を採り、B570 の同じ形の JSON
（Deno の双子 CLI `tools/anima-residency/profile.ts`）と key ごとに並べた。

## 1. 要約

- M2 の DiT 段は GPU 律速（GPU 時間 / 段の壁 = 96〜99%）。遅さは GPU のカーネルの中にあり、ホストとの往復ではない。
- f32 の GEMM / attention / conv の**大タイル**（128×128 など）が、B570 比で公称の性能比（3.2〜4.0 倍）の 2〜4 倍遅い。
  中タイル（64×32）と帯域律速の小カーネルは公称比どおり。**大タイルの幾何が M2 に合っていない**。
- 既定 quant の i8a8（dp4a）は M2 で f32 と同じ時間しかかからず、B570 比 29 倍になる。**a8 の利得が Metal では消える**
  （Tint が packed 内積を整数演算に展開するため — Codex 棚卸しの読み取り。M2 の命令列は未確認）。

## 2. 計測条件

| 項目          | M2                                                                             | B570                                                                            |
| ------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| 機体          | Apple M2（adapter `Apple M2` / `metal-3`・GPU コア数は JSON から判別できない） | Intel Arc B570（BMG G21）/ Linux                                                |
| 実行環境      | Chrome 153（Dawn / Metal）・確認ページ                                         | Deno 2.9.6（wgpu / Vulkan）・`profile.ts`                                       |
| checkout      | `536421fe`（dirty なし）                                                       | `c81b74fc`（dirty = per-op 記録の実装中）                                       |
| モデル        | `anima-turbo-v1.1`（8 step・CFG 1）・manifest `26f62c4e…`                      | 同じ                                                                            |
| quant         | `f16` と既定 `f16+dit8-a8-attn8-s16`                                           | 同じ                                                                            |
| 解像度 × 回数 | 512² × 2 → 1024² × 2（quant ごと・同じ pipeline）                              | 512² × 2（quant ごと）                                                          |
| 常駐          | `residency: "transformer"`（DiT のみ常駐・text / VAE は毎回構築）              | 同じ                                                                            |
| 時計の単位    | ns（`gpuTiming.unit: "ns"`）                                                   | raw tick → **52.08 ns/tick で換算**（known-issues の B570 / Deno の未換算問題） |
| ダミー        | 無し（`dummies.heldBytes = 0`）                                                | 無し                                                                            |

- 入力は既定 prompt・seed 42。DiT 段の run 数は 8（= 8 step）。
- M2 の timestamp は量子化されていない。456 entry のうち 100 µs の倍数も 65.536 µs の倍数も 0 本。負の標本も 0。
- GPU 時間を採ると 1 dispatch = 1 pass に開く。したがって本表の壁時計は、計測を切った走行の壁時計と比べられない。
- 比は M2 / B570。**GPU の差と実行系の差（Dawn / Metal と wgpu / Vulkan・シェーダコンパイラ）が混ざっている**。
  同じ実行系どうしの比ではない。
- 1 回目の行は初回の準備（DiT のロード・新しい解像度の初回）を含みうるので、以下は 2 回目の行（M2 の行 2 / 4 / 6 / 8・B570 の行 2）を代表に使う。

## 3. 段別の壁と GPU 時間

単位は秒。「壁」は段の開始から終了まで（text / VAE は Session の構築と解放を含む）。「GPU」は段の全 run の
`lastRunTiming.totalNs` の和。

| 機   | quant | 解像度 | 全体の壁 | text_encoder 壁 | text_conditioner 壁 | transformer 壁 | transformer GPU | GPU / 壁 | vae_decoder 壁 | vae_decoder GPU |
| ---- | ----- | ------ | -------: | --------------: | ------------------: | -------------: | --------------: | -------: | -------------: | --------------: |
| M2   | f16   | 512²   |    88.38 |            2.01 |                0.55 |          80.53 |           77.27 |      96% |           4.08 |            3.77 |
| M2   | f16   | 1024²  |   387.10 |            1.91 |                0.61 |         349.71 |          340.92 |      97% |          33.59 |           32.61 |
| M2   | 既定  | 512²   |    88.45 |            2.10 |                0.53 |          81.53 |           79.51 |      98% |           3.88 |            3.59 |
| M2   | 既定  | 1024²  |   399.67 |            2.18 |                0.58 |         363.18 |          358.41 |      99% |          33.12 |           32.19 |
| B570 | f16   | 512²   |    11.67 |            1.82 |                0.43 |           8.63 |            8.00 |      93% |           0.69 |            0.49 |
| B570 | 既定  | 512²   |     6.93 |            1.56 |                0.41 |           4.19 |            3.45 |      82% |           0.68 |            0.49 |

- DiT 段: M2 / B570 は f16 で GPU 9.7 倍、既定で GPU 23.1 倍。B570 では既定が f16 の 2.3 倍速いが、M2 では同じ
  （512² で 79.5 s 対 77.3 s）。1024² では既定のほうが遅い（358.4 s 対 340.9 s）。
- text 2 段: M2 の text_encoder 1.9〜2.2 s・conditioner 0.53〜0.61 s。B570 との壁の比は 1.1〜1.3 倍で、問題ではない
  （GPU 時間は段の壁の 6〜9%。残りは GPU 外で、Session の構築・解放を含む）。
- VAE: M2 は 512² で 3.9〜4.1 s（GPU 3.6〜3.8 s）、1024² で 33 s（GPU 32 s・タイル 9 枚）。
- 同じ条件の 1 回目との差: M2 の 512² は 93.62 / 92.07 s（行 1 / 5）対 88.38 / 88.45 s。1024² は 385.63 / 401.00 s
  （行 3 / 7）対 387.10 / 399.67 s。PNG の sha は quant × 解像度ごとに 2 回とも一致した。

### 3.1 先行実測の text 44 s との差（原因 = 重みをネットワーク越しに毎生成取得していた）

先行実測（`anima-residency-browser-2026-09-27T12-19-09.299Z.json`・checkout `915a6764`・GPU 時間なし）では、
text_encoder 43.6〜45.0 s・text_conditioner 11.6〜12.0 s・VAE 10.0〜10.3 s だった（4 行とも）。DiT は 2〜4 行目で
79.6 s。

- quant は今回の既定と同じ。先行の PNG sha（`0a5695470e4a…`）が今回の行 5 / 6 と一致する。
- **ダミーは原因ではない。** 先行の行 1〜3 は `dummyBytesHeld = 0`（ダミーの確保は 12:15:56 以降 = 行 3 の後）で、
  80 GiB を保持した行 4 も text 44.06 s だった。ダミーの有無で text の時間は変わっていない。
- `915a6764..536421fe` の差は `tools/` と `docs/` だけで、`packages/`（runtime / models）は同じコード。
- text / VAE はどちらの走行でも generate ごとに Session を作り直す。今回はそれが 2 s で済んだので、Codex 棚卸しの
  候補 ③（毎回の重み取得・構築と Chrome のホスト往復）は**恒常的な原因ではない**。
- **原因は取得経路**（利用者の確認 2026-09-27）: 先行走行は Linux 機の確認ページをポート転送で開き、重みをその都度
  ネットワーク越しに取得していた。今回の走行はリポとミラーを Mac へ pull してローカルで配った。text / VAE は generate
  ごとに Session を作り直す（`residency: "transformer"` は DiT だけ保持）ので、供給が遅い経路では毎回 text 1.1 GiB 級の
  取得が壁に乗る（44 s ≈ 25 MB/s）。記録に `buildStats` が無いので内訳（HTTP / 検証 / アップロード）は取れていない。
- 含意: 配布形を HF から直接読むブラウザ利用でも、Chrome の HTTP キャッシュが Range 応答を保持しない限り同じ形に
  なる。「text / VAE の Session（または取得済みバイト列）を生成間で保持する」は別起票の候補（§7 (e)）。

## 4. key 別の比（M2 / B570・512²）

「µs/disp」は key の合計時間 ÷ dispatch 本数。同じ key に複数の形が合流するので、形ごとの効率ではない。
各段の上位の key（段の GPU 時間の大半）を載せる。

### 4.1 quant `f16`（M2 の行 2 対 B570 の行 2）

| 段               | key                                             |    n |   M2 ms | B570 ms | M2 µs/disp | B570 µs/disp |    比 |
| ---------------- | ----------------------------------------------- | ---: | ------: | ------: | ---------: | -----------: | ----: |
| text_encoder     | `linear:v2:f32:reg16x16r1x4w4v4:wf16`           |  196 |   119.7 |    56.8 |        611 |          290 |  2.1× |
| text_conditioner | `linear:v2:f32:reg16x16r1x4w4v4:wf16`           |   61 |    42.8 |    17.6 |        702 |          289 |  2.4× |
| transformer      | `linear:v2:f32:reg128x128r8x8w16v4:wf16`        | 1808 | 65449.0 |  6429.3 |      36200 |         3556 | 10.2× |
| transformer      | `attention_qk:v1:f32:reg128x128r8x8w16v4`       |  448 |  3703.7 |   273.9 |       8267 |          611 | 13.5× |
| transformer      | `attention_pv:v1:f32:reg128x128r8x8w16v4`       |  448 |  2772.5 |   282.3 |       6189 |          630 |  9.8× |
| transformer      | `linear:v2:f32:reg64x32r4x4w8v4:wf16`           |  448 |  1502.5 |   340.0 |       3354 |          759 |  4.4× |
| transformer      | `strided:v1:f32:r4:wg256`                       | 3832 |  1067.8 |   102.3 |        279 |           27 | 10.4× |
| transformer      | `attention_stats:v2:f32:lastdim:safe:wg256:rc4` |  224 |   421.2 |   104.9 |       1880 |          468 |  4.0× |
| transformer      | `ew:v3:gelu:f32>f32:r3:wg256`                   |  224 |   411.3 |    49.5 |       1836 |          221 |  8.3× |
| transformer      | `ew:v3:mul:f32>f32:r3:wg256`                    |  672 |   380.5 |    33.6 |        566 |           50 | 11.3× |
| transformer      | `attention_stats:v2:f32:lastdim:safe:wg256:rc2` |  224 |   354.6 |    86.2 |       1583 |          385 |  4.1× |
| transformer      | `rms_norm:v1:f32:lastdim:wg128`                 |  896 |   350.5 |    85.0 |        391 |           95 |  4.1× |
| transformer      | `ew:v3:add:f32>f32:r3:wg256`                    |  672 |   348.8 |    59.5 |        519 |           88 |  5.9× |
| transformer      | `adaln_norm:v1:lastdim:f32:wg256`               |  680 |   240.8 |    64.1 |        354 |           94 |  3.8× |
| transformer      | `rope:v1:half:f32:wg256`                        |  448 |   171.1 |    32.8 |        382 |           73 |  5.2× |
| transformer      | `linear_gemv:v1:f32:c32u4:wf16`                 | 1376 |    94.3 |    51.4 |         69 |           37 |  1.8× |
| vae_decoder      | `conv2d:v3:f32:igemm64x128v4:wg16x8:wf16`       |   28 |  2007.5 |   227.2 |      71697 |         8114 |  8.8× |
| vae_decoder      | `conv2d:v3:f32:igemm32x128v4:wg16x4:wf16`       |    9 |  1182.8 |   192.4 |     131425 |        21376 |  6.1× |
| vae_decoder      | `ew:v3:mul:f32>f32:r4:wg256`                    |   90 |   282.3 |    25.0 |       3136 |          278 | 11.3× |
| vae_decoder      | `ew:v3:div:f32>f32:r4:wg256`                    |   30 |    93.6 |     8.5 |       3121 |          283 | 11.0× |
| vae_decoder      | `reduce:v2:sum:f32>f32:axis:wg256`              |   30 |    60.5 |    11.3 |       2016 |          377 |  5.3× |
| vae_decoder      | `ew:v3:add:f32>f32:r4:wg256`                    |   15 |    42.3 |     5.6 |       2823 |          372 |  7.6× |
| vae_decoder      | `attention_qk:v1:f32:reg128x128r8x8w16v4`       |    1 |    31.2 |     2.9 |      31151 |         2941 | 10.6× |
| vae_decoder      | `silu:v1:x-sigmoid:f32:wg256`                   |   29 |    31.1 |     7.9 |       1072 |          271 |  4.0× |
| vae_decoder      | `attention_pv:v1:f32:reg128x128r8x8w16v4`       |    1 |    24.6 |     3.6 |      24553 |         3550 |  6.9× |

### 4.2 既定 quant `f16+dit8-a8-attn8-s16`（M2 の行 6 対 B570 の行 2）

| 段               | key                                                      |    n |   M2 ms | B570 ms | M2 µs/disp | B570 µs/disp |    比 |
| ---------------- | -------------------------------------------------------- | ---: | ------: | ------: | ---------: | -----------: | ----: |
| text_encoder     | `linear:v2:f32:reg16x16r1x4w4v4:wf16`                    |  196 |   119.7 |    56.4 |        610 |          288 |  2.1× |
| text_conditioner | `linear:v2:f32:reg16x16r1x4w4v4:wf16`                    |   61 |    37.0 |    15.7 |        607 |          258 |  2.3× |
| transformer      | `linear:v4:i8a8:tile128x64r8x8w8x16k16v4:dp4a`           | 3632 | 67894.5 |  2369.6 |      18693 |          652 | 28.7× |
| transformer      | `attention_pv:v3:i8a8:tile64x128r8x8w16x8k16v4:dp4a:s16` |  448 |  3593.3 |   143.9 |       8021 |          321 | 25.0× |
| transformer      | `attention_qk:v3:i8a8:tile128x64r8x8w8x16k16v4:dp4a:s16` |  448 |  3568.3 |   105.7 |       7965 |          236 | 33.8× |
| transformer      | `strided:v1:f32:r4:wg256`                                | 4280 |  1170.3 |   131.2 |        273 |           31 |  8.9× |
| transformer      | `quantize_rows:v1:f32>i8:pertoken:wg256`                 | 3168 |   623.5 |   131.5 |        197 |           42 |  4.7× |
| transformer      | `ew:v3:gelu:f32>f32:r3:wg256`                            |  224 |   393.6 |    49.5 |       1757 |          221 |  7.9× |
| transformer      | `attention_stats:v2:f32:lastdim:safe:wg256:s16:rc4`      |  224 |   385.9 |   105.6 |       1723 |          471 |  3.7× |
| transformer      | `attention_stats:v2:f32:lastdim:safe:wg256:s16:rc2`      |  224 |   341.6 |    85.7 |       1525 |          382 |  4.0× |
| transformer      | `rms_norm:v1:f32:lastdim:wg128`                          |  896 |   329.6 |    84.9 |        368 |           95 |  3.9× |
| transformer      | `ew:v3:add:f32>f32:r3:wg256`                             |  672 |   326.4 |    60.1 |        486 |           89 |  5.4× |
| transformer      | `ew:v3:mul:f32>f32:r3:wg256`                             |  672 |   322.2 |    32.9 |        480 |           49 |  9.8× |
| transformer      | `adaln_norm:v1:lastdim:f32:wg256`                        |  680 |   232.6 |    61.9 |        342 |           91 |  3.8× |
| transformer      | `rope:v1:half:f32:wg256`                                 |  448 |   159.1 |    32.8 |        355 |           73 |  4.8× |
| transformer      | `quantize_rows:v1:f32>i8:pertoken:wg256:r8w32`           |  904 |   142.4 |    39.1 |        158 |           43 |  3.6× |
| transformer      | `quantize_rows:v1:f32>i8:pertoken:wg256:r2w128`          |  224 |    22.1 |     5.5 |         98 |           25 |  4.0× |
| vae_decoder      | `conv2d:v3:f32:igemm64x128v4:wg16x8:wf16`                |   28 |  1899.1 |   226.4 |      67826 |         8084 |  8.4× |
| vae_decoder      | `conv2d:v3:f32:igemm32x128v4:wg16x4:wf16`                |    9 |  1150.5 |   192.3 |     127835 |        21367 |  6.0× |
| vae_decoder      | `ew:v3:mul:f32>f32:r4:wg256`                             |   90 |   265.9 |    25.0 |       2955 |          278 | 10.6× |
| vae_decoder      | `ew:v3:div:f32>f32:r4:wg256`                             |   30 |    89.8 |     8.5 |       2992 |          282 | 10.6× |
| vae_decoder      | `reduce:v2:sum:f32>f32:axis:wg256`                       |   30 |    56.0 |    11.3 |       1865 |          376 |  5.0× |
| vae_decoder      | `ew:v3:add:f32>f32:r4:wg256`                             |   15 |    41.3 |     5.6 |       2754 |          372 |  7.4× |
| vae_decoder      | `attention_qk:v1:f32:reg128x128r8x8w16v4`                |    1 |    30.1 |     2.8 |      30050 |         2809 | 10.7× |
| vae_decoder      | `silu:v1:x-sigmoid:f32:wg256`                            |   29 |    27.4 |     7.9 |        943 |          271 |  3.5× |
| vae_decoder      | `attention_pv:v1:f32:reg128x128r8x8w16v4`                |    1 |    22.9 |     3.5 |      22897 |         3547 |  6.5× |

### 4.3 読み方

- **f32 の大タイル**: linear `reg128x128r8x8w16v4` 10.2 倍・attention qk 13.5 倍 / pv 9.8 倍・VAE の conv2d
  `igemm64x128` 8.4〜8.8 倍 / `igemm32x128` 6.0〜6.1 倍。f16 の DiT 段 GPU 時間の 84.7% が大タイルの linear 1 key。
- **f32 の中タイル**: linear `reg64x32r4x4w8v4`（cross attention の K/V・M = 512）4.4 倍。
- **i8a8（dp4a）**: linear 28.7 倍・attention qk 33.8 倍 / pv 25.0 倍。
- **帯域律速の小カーネル**: quantize_rows 3.6〜4.7 倍・attention_stats 3.7〜4.1 倍・rms_norm 3.9〜4.1 倍・adaln_norm 3.8 倍。
- **elementwise 系**: ew mul 9.8〜11.3 倍・gelu 7.9〜8.3 倍・strided 8.9〜10.4 倍・ew add 5.4〜5.9 倍・rope 4.8〜5.2 倍。
- **小 M の linear**（text 段・`reg16x16r1x4w4v4`）: 2.1〜2.4 倍。M = 1 の `linear_gemv` 1.8 倍。

## 5. 公称の性能比と達成効率

### 5.1 公称の性能比（各社の公開値）

|            | B570（公称）   | M2（公称）                              |   B570 / M2 |
| ---------- | -------------- | --------------------------------------- | ----------: |
| GPU        | 18 Xe コア     | 8 または 10 コア（本機は未確認）        |           — |
| FP32       | 約 11.5 TFLOPS | 約 2.9（8 コア）〜 3.6（10 コア）TFLOPS | 3.2〜4.0 倍 |
| メモリ帯域 | 380 GB/s       | 100 GB/s                                |      3.8 倍 |

§4 の比を公称比で割った「公称比を超える倍率」:

| 群                                                                           |      実測比 | 基準にした公称比 |      超過 |
| ---------------------------------------------------------------------------- | ----------: | ---------------- | --------: |
| 帯域律速の小カーネル（quantize_rows・attention_stats・rms_norm・adaln_norm） |    3.6〜4.7 | 帯域 3.8         |  0.9〜1.2 |
| f32 中タイル linear（`reg64x32`）                                            |         4.4 | 演算 3.2〜4.0    |  1.1〜1.4 |
| f32 大タイル linear（`reg128x128`）                                          |        10.2 | 演算 3.2〜4.0    |  2.6〜3.2 |
| f32 attention qk / pv（大タイル）                                            |  13.5 / 9.8 | 演算 3.2〜4.0    |  2.4〜4.2 |
| f32 conv2d igemm（VAE）                                                      |    6.0〜8.8 | 演算 3.2〜4.0    |  1.5〜2.8 |
| elementwise 系（mul・gelu・strided）                                         |   7.9〜11.3 | 帯域 3.8         |  2.1〜3.0 |
| i8a8 linear                                                                  |        28.7 | 演算 3.2〜4.0    |  7.2〜9.0 |
| i8a8 attention qk / pv                                                       | 33.8 / 25.0 | 演算 3.2〜4.0    | 6.3〜10.6 |

帯域律速の小カーネルと中タイルは、ほぼ公称比どおり。M2 側の遅さはハードの差だけで説明できる。大タイルと i8a8 は
公称比を大きく超える。カーネル側が M2 に合っていない。

### 5.2 達成効率（推定）

仕事量の根拠: anima DiT の linear は S = 4096（1024²）で 6,795.94 GMAC。内訳は encoder 側 60.13 G・M = 1 が 0.195 G
（`.claude/reviews/2026-09-25_perf-recon/deep/B4-hoist-cfg-h30.md` §2.4 のグラフ再集計）。S に比例する残りを 1/4 にして
512²（S = 1024）へ換算すると、1 forward ≈ 1.74 TMAC = 3.5 TFLOP。8 step で 27.9 TFLOP。

|                                               | linear の GPU 時間（8 step） |            達成 | 公称に対する割合 |
| --------------------------------------------- | ---------------------------: | --------------: | ---------------: |
| M2・f32 計算（`f16` の linear 3 key の和）    |                       67.0 s |  約 0.42 TFLOPS |          12〜14% |
| B570・f32 計算（同上）                        |                       6.82 s |   約 4.1 TFLOPS |           約 36% |
| M2・i8a8（`linear:v4:i8a8` 1 key・全 linear） |                       67.9 s |      f32 と同等 |    a8 の利得 ≈ 0 |
| B570・i8a8（同上）                            |                       2.37 s | f32 の 2.9 倍速 |                — |

- 「linear 3 key」= 大タイル・中タイル・`linear_gemv`。既定 quant ではこの 3 群が全部 `linear:v4:i8a8` 1 key に入る
  （本数 1,808 + 448 + 1,376 = 3,632）。
- 数値は推定。MAC 数はグラフの静的集計で、M2 の GPU コア数も未確認。i8a8 の行には activation の量子化
  （`quantize_rows`）の時間は入っていない。

## 6. 帰属

Codex 棚卸しの候補 ① と ② が実測で支持される。

1. **f32 の GEMM / attention / conv の大タイル幾何が M2 に合っていない（候補 ②）。** 128×128 タイル・1 スレッド 64 累積・
   wg256 の key が公称比の 2〜4 倍遅い。同じ f32 GEMM でも中タイル（64×32）は公称比並み。この差が、幾何が主因である
   ことの直接の証拠になる。実行系の差（Dawn / Metal 対 wgpu / Vulkan）がすべての key に一様に効くなら、中タイルと
   小カーネルも同じだけ超過するはずだが、そうなっていない。
2. **i8a8（dp4a）の経路は Metal で利得を失う（候補 ①）。** M2 では i8a8 の linear が f32 計算と同じ時間になり、公称比の
   7〜9 倍遅い。Tint が `dot4I8Packed` を unpack と整数の乗算・加算に展開する（Codex 棚卸し F3 のソース確認）ことと
   整合する。幾何を直しても、この問題は別に残る。
3. **elementwise 系の超過（公称比の 2〜3 倍）は副次。** 該当 key（strided・gelu・mul・add・rope）の合計は f16 の DiT
   段 GPU 時間の 3.1%（2.38 s / 77.27 s）。
4. **text 2 段は GPU の問題ではない。** 先行実測の 44 s はネットワーク越しの重み取得で、ローカル配信では 2 s（§3.1）。

M5 の「1024² 全体 80 s」は利用者の記憶だけで、同条件の記録ではない。本帰属は M2 だけに基づく。

## 7. 次の一手（候補・裁定待ち）

- (a) **M2 で GEMM 幾何を掃引する。** 大 M の linear / attention / conv2d について、64×64・64×32・wg128 などを同じ形で
  比べる。Deno / Metal は timestamp が取れない（limitations「Metal（Apple GPU）では GPU 側 timestamp 計測が実用に
  ならない」）ので、Chrome で回す掃引ページ（`tools/llm-speed/browser` の形）が要る。
  → **裁定 a（2026-09-27）・道具は `tools/geometry-sweep`（Deno CLI + Chrome ページ port 8789・形式
  `karume-geometry-sweep/2`・commit `c562c7ae`）**。runtime の codegen 入口に明示の幾何を足し（`9947568c`・省略時は
  バイト同一）、同じ shape で幾何だけを変えて timestamp で測り、既定幾何との出力 sha256 の一致も記録する。M2 の実測は
  利用者の Chrome で採る（`tools/geometry-sweep/browser/README.md`）。
- (b) **Metal では既定 quant の a8 を外す**（i8 重み × f32 計算 `:wi8`）。幾何が直っても i8a8 は別の問題として残る。
  manifest には `f16+dit8`（i8 重み × f32 計算）の quant があるので、確認ページのままで M2 の実測が採れる。
- (c) **B570 でも中タイル・大タイルの効率を見直す余地がある。** 達成 36% は高くない（K-67 の再実測と重なる）。
  B570 の quick 掃引（`outputs/bench/karume/2026-09-27_geometry-sweep/quick-linear-after-fix.json`・raw tick・既定の再測定
  0.95〜1.02 の 11 / 14 ケース）: 大 M（1024 / 4096）の f32 linear と i8a8 linear は既定幾何が最速（他は ×0.52〜0.97）。
  中 M 512（既定 64×32）では 128×128 が ×1.41、小 M 64（既定 16×16）では 64×32 が ×1.49・64×64 が ×1.48 — K-67 の材料。
  全 77 行で幾何間の出力 sha256 は一致。
- (d) 供給側の内訳が採れるよう、確認ページの記録に Session の `buildStats` を足す（§3.1）。
- (e) text / VAE の重みを生成間で保持する（Session の常駐か取得済みバイト列の保持）— ネットワーク供給の 44 s / 10 s は
  DiT の次に大きい壁（§3.1）。メモリとの引き換えなので別起票で裁定する。

## 8. 未確認

- M2 の GPU コア数（8 / 10）。公称比に 3.2〜4.0 の幅があるのはこのため。
- Chrome 153 に入っている Dawn / Tint の revision。Codex の F3 は Dawn `6c1e2771` のソースで確かめたもの。
- M2 の Metal コンパイラが出す最終的な命令列（packed 内積の命令が本当に無いか）。
- 幾何以外の要因（レジスタ spill・占有率・threadgroup メモリ）。(a) の掃引で切り分ける。
- M5 の値（利用者の記憶の 80 s のみ）。
- 先行実測の text 44 s / VAE 10 s の内訳（HTTP 取得・検証・アップロードの比 — §3.1。経路の違いは利用者確認済み）。
- 同じ実行系どうしの比（B570 を Chrome で回した値）。本表は GPU の差と実行系の差が混ざった比。

## 9. 幾何掃引の結果（M2 quick・2026-09-27 16:28）

`tools/geometry-sweep` の Chrome ページを利用者の M2（Chrome 153 / metal-3・developer features 有効・timestamp は非量子化）で
quick（op 5 族・33 ケース・177 行・rounds 5）で回した。JSON は `outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json`。
失敗 0・**幾何間の出力 sha256 は全 177 行で既定と一致**（f32 も i8a8 も幾何でビットは動かない）・既定の再測定は全ケース 0.99〜1.01。
「同幾何の B570 比」は B570 の quick（linear / i8a8-linear のみ・raw tick）の同じ幾何の対既定。

| case                               | 既定幾何               | M2 既定 µs/disp | M2 TFLOPS | M2 最良幾何            | 対既定 |                           同幾何の B570 比 | 既定再測定 |
| ---------------------------------- | ---------------------- | --------------: | --------: | ---------------------- | -----: | -----------------------------------------: | ---------: |
| linear-m1024-n2048-k2048           | reg128x128r8x8w16      |           19694 |      0.44 | reg64x32r4x4w8         |  ×1.58 |                                      ×0.78 |      ×1.00 |
| linear-m1024-n8192-k2048           | reg128x128r8x8w16      |           78673 |      0.44 | reg64x32r4x4w8         |  ×1.59 |                                      ×0.74 |      ×1.01 |
| linear-m1024-n2048-k8192           | reg128x128r8x8w16      |           79499 |      0.43 | reg64x32r4x4w8         |  ×1.59 |                                      ×0.76 |      ×1.01 |
| linear-m4096-n2048-k2048           | reg128x128r8x8w16      |           78817 |      0.44 | reg64x32r4x4w8         |  ×1.59 |   ×0.98（B570 最良 reg64x64r4x4w16 ×1.21） |      ×1.01 |
| linear-m4096-n8192-k2048           | reg128x128r8x8w16      |          315760 |      0.44 | reg64x32r4x4w8         |  ×1.60 |                                      ×0.72 |      ×1.00 |
| linear-m4096-n2048-k8192           | reg128x128r8x8w16      |          319085 |      0.43 | reg64x32r4x4w8         |  ×1.60 |                                      ×0.74 |      ×1.00 |
| linear-m512-n2048-k1024            | reg64x32r4x4w8         |            3184 |      0.67 | reg64x32r4x4w8（既定） |  ×1.00 | ×1.00（B570 最良 reg128x128r8x8w16 ×1.41） |      ×1.00 |
| linear-m64-n3072-k1024             | reg16x16r1x4w4         |            1418 |      0.28 | reg64x64r4x4w16        |  ×2.27 |    ×1.48（B570 最良 reg64x32r4x4w8 ×1.49） |      ×1.00 |
| i8a8-linear-m1024-n2048-k2048      | tile128x64r8x8w8x16k16 |           20468 |      0.42 | tile64x64r8x4w16x8k16  |  ×1.06 |                                      ×0.83 |      ×1.00 |
| i8a8-linear-m1024-n8192-k2048      | tile128x64r8x8w8x16k16 |           80518 |      0.43 | tile64x64r8x4w16x8k16  |  ×1.12 |                                      ×0.80 |      ×1.00 |
| i8a8-linear-m1024-n2048-k8192      | tile128x64r8x8w8x16k16 |           81677 |      0.42 | tile64x64r8x4w16x8k16  |  ×1.13 |                                      ×0.84 |      ×1.01 |
| i8a8-linear-m4096-n2048-k2048      | tile128x64r8x8w8x16k16 |           80660 |      0.43 | tile64x64r8x4w16x8k16  |  ×1.12 |                                      ×0.81 |      ×1.00 |
| i8a8-linear-m4096-n8192-k2048      | tile128x64r8x8w8x16k16 |          322485 |      0.43 | tile64x64r8x4w16x8k16  |  ×1.12 |                                      ×0.64 |      ×1.00 |
| i8a8-linear-m4096-n2048-k8192      | tile128x64r8x8w8x16k16 |          325153 |      0.42 | tile64x64r8x4w16x8k16  |  ×1.12 |                                      ×0.82 |      ×1.00 |
| attention-qk-self-m1024-n1024      | reg128x128r8x8w16      |           10447 |      0.41 | reg64x32r4x4w8         |  ×1.58 |                                          - |      ×1.00 |
| attention-pv-self-m1024-n1024      | reg128x128r8x8w16      |            7946 |      0.54 | reg64x64r8x4w16        |  ×1.66 |                                          - |      ×0.99 |
| attention-qk-self-m4096-n4096      | reg128x128r8x8w16      |          162664 |      0.42 | reg64x32r4x4w8         |  ×1.55 |                                          - |      ×1.01 |
| attention-pv-self-m4096-n4096      | reg128x128r8x8w16      |          126330 |      0.54 | reg64x64r8x4w16        |  ×1.67 |                                          - |      ×1.00 |
| attention-qk-cross-m1024-n512      | reg128x128r8x8w16      |            5304 |      0.40 | reg64x32r4x4w8         |  ×1.60 |                                          - |      ×1.00 |
| attention-pv-cross-m1024-n512      | reg128x128r8x8w16      |            4022 |      0.53 | reg64x64r8x4w16        |  ×1.67 |                                          - |      ×1.01 |
| attention-qk-cross-m4096-n512      | reg128x128r8x8w16      |           20771 |      0.41 | reg64x32r4x4w8         |  ×1.57 |                                          - |      ×1.01 |
| attention-pv-cross-m4096-n512      | reg128x128r8x8w16      |           15765 |      0.54 | reg64x64r8x4w16        |  ×1.66 |                                          - |      ×1.00 |
| i8a8-attention-qk-self-m1024-n1024 | tile128x64r8x8w8x16k16 |           10599 |      0.41 | tile64x64r8x4w16x8k16  |  ×1.13 |                                          - |      ×1.00 |
| i8a8-attention-pv-self-m1024-n1024 | tile64x128r8x8w16x8k16 |           10648 |      0.40 | tile64x64r8x4w16x8k16  |  ×1.10 |                                          - |      ×1.00 |
| i8a8-attention-qk-self-m4096-n4096 | tile128x64r8x8w8x16k16 |          169757 |      0.40 | tile64x64r8x4w16x8k16  |  ×1.13 |                                          - |      ×1.00 |
| i8a8-attention-pv-self-m4096-n4096 | tile64x128r8x8w16x8k16 |          168201 |      0.41 | tile64x64r8x4w16x8k16  |  ×1.09 |                                          - |      ×1.00 |
| i8a8-attention-qk-cross-m1024-n512 | tile128x64r8x8w8x16k16 |            5309 |      0.40 | tile64x64r8x4w16x8k16  |  ×1.13 |                                          - |      ×1.00 |
| i8a8-attention-pv-cross-m1024-n512 | tile64x128r8x8w16x8k16 |            5365 |      0.40 | tile64x64r8x4w16x8k16  |  ×1.11 |                                          - |      ×1.00 |
| i8a8-attention-qk-cross-m4096-n512 | tile128x64r8x8w8x16k16 |           21222 |      0.40 | tile64x64r8x4w16x8k16  |  ×1.13 |                                          - |      ×1.00 |
| i8a8-attention-pv-cross-m4096-n512 | tile64x128r8x8w16x8k16 |           21138 |      0.41 | tile64x64r8x4w16x8k16  |  ×1.09 |                                          - |      ×1.00 |
| conv2d-c96-512x512                 | igemm32x128:wg16x4     |          139538 |      0.31 | igemm64x64:wg16x8      |  ×1.48 |                                          - |      ×0.99 |
| conv2d-c192-256x256                | igemm64x128:wg16x8     |          103976 |      0.42 | igemm64x64:wg16x16     |  ×1.37 |                                          - |      ×1.00 |
| conv2d-c384-128x128                | igemm64x128:wg16x8     |          122309 |      0.36 | igemm64x64:wg16x16     |  ×1.43 |                                          - |      ×1.00 |

読み:

- **f32 GEMM の大 M（DiT の linear）は、M2 では 64×32（`reg64x32r4x4w8`・128 スレッド・1 スレッド 16 累積）が既定 128×128 の
  1.58〜1.60 倍。** 64×64（r8x4 w16x8 / r4x4 w16x16）も 1.52〜1.58 倍で、大タイルだけが外れている。B570 では同じ幾何が
  0.72〜0.98 倍で既定が最速 — **勝つ幾何が GPU で逆転する**。
- f32 attention も同じ: ①QK は 64×32 が 1.55〜1.60 倍、③PV は 64×64（r8x4 w16x8）が 1.66〜1.67 倍。
- conv2d（VAE）は 64×64 が 1.37〜1.48 倍（Cout=96 の 32×128 既定・Cout=192 / 384 の 64×128 既定とも）。
- **i8a8 は幾何ではほぼ動かない**（最良 tile64x64 r8x4 で 1.06〜1.13 倍）。M2 の i8a8 既定 20.5 ms/dispatch は、同 shape の f32 既定
  19.7 ms と同じで、f32 を 64×32 にすれば 12.5 ms — Metal では a8 は幾何を直しても f32 に負ける（dp4a の展開が主因のまま）。
- 小 M 64（text 段）は M2 で 64×64 が 2.27 倍、B570 でも 64×32 が 1.49 倍 — 既定 16×16 は両 GPU で外れている（K-67）。
- 達成効率: M2 の f32 既定 0.44 TFLOPS → 64×32 で約 0.70 TFLOPS（公称 2.9〜3.6 の 20〜24%）。B570 の 36% にはまだ届かず、
  full の格子（f32 54 幾何）で更に上がる余地がある。

含意（裁定待ち）: 幾何の選択を「shape の純関数」から「shape × adapter の静的な表」（vendor `apple` なら Metal 向けの表）へ
広げれば、M2 の DiT（f16 quant）は linear 85% × 1.6 倍 + attention で **約 1.5 倍**の見込み。既定 quant（i8a8）は Metal では
f32 計算（i8 重み `:wi8`）へ落とすのが速い。ADR 0022 の MUST（実行時オートチューン禁止・キーに幾何判別子）は表が静的なら保てる。

## 10. full 掃引と裁定（2026-09-27 追記）

**M2 の full**（`outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json`・linear / attention / conv2d の
格子全体・19 ケース・927 行）: 失敗 0・**全 927 行で出力 sha256 が既定と一致**。ケースごとの最良幾何:

- linear の大 M（6 ケース）: `reg128x32r8x4w8` が ×1.68〜1.77。quick の最良 64×32 より上。
- linear の M = 512: `reg128x32r8x4w8` ×1.10。M = 64: `reg64x64r4x4w16` ×2.26。
- attention ①QK（4 ケース）: `reg128x32r8x4w8` が ×1.57〜1.83。
- attention ③PV: `reg32x64r4x8w8` が ×1.36〜1.69。cross m1024-n512 だけは `reg64x128r4x8w16` ×1.18 が最良。
- conv2d: c96 / c384 は `igemm128x64:wg16x16` が ×1.65 / ×1.71。c192 は `igemm64x64:wg16x16` ×1.36。
- ③PV の 2 ケースで、ケース末尾の既定の再測定比が 1.1 を超えた（pv-self-m1024 ×1.228・pv-cross-m1024 ×1.126）。
  この 2 ケースの比は機の揺れを含む。

quick と full を生成規則（ADR 0115 §4 — クラスの全ケースで ×1.05 以上・幾何平均が最大）で合成した結果が
`apple-metal-3` で、採用表は ADR 0115 Consequences が正本。③PV はクラスの全ケースで勝つ幾何として `reg64x64r8x4w16`
（×1.49）が残った。

**RTX 5070 Ti の full**（`outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json`・Windows / Chrome 153・
GPU timestamp ns・op 5 族・33 ケース・1599 行・既定の再測定比 0.992〜1.011）: 公開 Artifact（静的配信のページ）から
取得した。Artifact の iframe で WebGPU が動く。

- f32 の大 M（≥ 1024）は既定 `reg128x128r8x8w16` が最速（代替は ×0.93〜1.004）。
- M = 512 は `reg128x128r8x8w16` が ×1.475（B570 の quick の ×1.41 と同じ傾向）。M = 64 は `reg32x32r2x4w8` が ×1.352
  （B570 は 64×32 が ×1.49）。
- attention ①QK は既定が最速（代替 ×0.92〜0.96）。③PV は `reg64x128r8x8w16` が ×1.02〜1.05 で、門の ×1.05 に届かない。
- conv2d は既定が最速か僅差（c384 の `igemm128x128:wg16x16` ×1.092 だけ）。
- **i8a8 は 3 欄とも `tile128x64r8x4w16x16k16` が全ケースで勝つ**: linear ×1.156〜1.256・①QK ×1.199〜1.238・
  ③PV ×1.134〜1.306。

**i8a8 ①QK の 16 幾何の不一致は誤値（同日に修正）**: RTX の full で、i8a8-attention の qk ケースが 16 幾何で既定と出力不一致に
なった（B570 でも同じ 16 幾何）。原因は生成器の変数のシャドーイング（K 側の充填スロット 5 以上で `var k4` が K のパック数を隠す）で、
本番の幾何は影響外。修正は CHANGELOG（Unreleased の Fixed）。この掃引 JSON の 16 幾何の速度値は無効（ロードを飛ばしている）。

**裁定 → [ADR 0115](../decisions/0115-geometry-profiles.md)**: `apple-metal-3` を登録（perf-ledger K-71）。
nvidia-blackwell プロファイルの生成（perf-ledger K-67）と i8a8 ①QK の修正は裁定待ち。「Metal で既定 quant の a8 を外す」は
保留で、既定の quant 席を量子化にするか opt-in にするかの再検討に合流した。

## 11. プロファイル適用後の M2 再測（2026-09-29 追記・時点スナップショット）

利用者の M2（Chrome 153・`apple` / `metal-3`）で確認ページ（`deno task bench:anima-browser`）を 512²・seed 42・DiT 常駐で
回した（checkout `cb87ab93`・記録 `outputs/bench-browser/anima-residency-browser-f16-2026-09-29T17-15-47.294Z.json`）。
2 回目以降の値。比較元は §3 / §9 の 2026-09-27 の 2 本。

| 席（512²）        | 項目             | 2026-09-27（既定幾何） | 2026-09-29（`apple-metal-3`） |            比 |
| ----------------- | ---------------- | ---------------------: | ----------------------------: | ------------: |
| `f16`             | DiT 段（壁時計） |                 80.5 s |                        45.1 s |         ×1.78 |
| `f16`             | DiT 段（GPU）    |                 77.3 s |                        44.2 s |         ×1.75 |
| `f16`             | 内訳 linear      |                 65.4 s |                        37.1 s |         ×1.76 |
| `f16`             | 内訳 ①QK / ③PV   |            3.7 / 2.8 s |                   2.0 / 1.6 s | ×1.85 / ×1.75 |
| 既定（i8a8・s16） | DiT 段（壁時計） |                 81.5 s |                        71.1 s |         ×1.15 |
| 既定（i8a8・s16） | DiT 段（GPU）    |                 79.5 s |                        69.9 s |         ×1.14 |
| 両席              | VAE 段           |             3.9〜4.4 s |                         2.9 s |          ×1.4 |
| 両席              | PNG sha256       |                   同じ |                          同じ |          一致 |

- §9 の見込み（linear ×1.74・attention ×1.5〜1.7 → DiT 段 約 1.5 倍）に対し、f16 quant の実測は ×1.78。
- 既定 quant（a8）は M2 では f16 quant より 1.58 倍遅い（71.1 s 対 45.1 s）。§9 の「a8 の利得ゼロ」は、f32 経路の幾何が直った
  ぶん「損」に変わった。判断は量子化 opt-in の再検討へ（backlog now）。
- text_encoder の linear は当時の ≤ 64 の規則 `reg64x64r4x4w16` で走っている（ADR 0115 追記決定 3 で既定へ戻った）。
  text 段 1.9 s のうち linear は 0.13 s で、DiT 段の比較には影響しない。

## 12. 3 機の f16 / 既定 quant 比較（2026-09-29 追記・時点スナップショット）

512²・seed 42・DiT 常駐の 2 回目以降・「既定」= `f16+dit8-a8-attn8-s16`（i8 重み + a8 整数内積 + s16）。B570 は §3 の
2026-09-27 の値（Deno・既定プロファイル）、M2 は §11、RTX 5070 Ti は
`outputs/bench-browser/anima-residency-browser-f16-2026-09-29T18-11-14.451Z.json`（Chrome・`nvidia-blackwell`・ADR 0115 追記 5）。

| 機（実行系・プロファイル）                       | f16: DiT 壁 / GPU | 既定: DiT 壁 / GPU |    既定 ÷ f16（GPU） | DiT の linear f32 / i8a8 |                       全体の壁 f16 / 既定 |
| ------------------------------------------------ | ----------------: | -----------------: | -------------------: | -----------------------: | ----------------------------------------: |
| Intel Arc B570（Deno・既定幾何・09-27）          |     8.63 / 8.00 s |      4.19 / 3.45 s |           ×2.32 速い |              6.2 / 2.6 s |                            11.67 / 6.93 s |
| Apple M2（Chrome・`apple-metal-3`・09-29）       |     45.1 / 44.2 s |      71.1 / 69.9 s | ×0.63（1.58 倍遅い） |            37.1 / 59.1 s |                             50.4 / 76.5 s |
| RTX 5070 Ti（Chrome・`nvidia-blackwell`・09-29） |     2.86 / 2.79 s |      1.21 / 1.14 s |           ×2.45 速い |            2.31 / 0.74 s | 19.4 / 17.9 s（text 12.8 s はポート転送） |

- 整数内積（a8・`dot4I8Packed`）が効く機では既定 quant が f16 の 2.3〜2.5 倍速い（B570 = Xe2 の DP4A・RTX = Blackwell）。
  カーネル単位の f32 / i8a8 比は B570 2.4〜4.1・RTX 2.7〜3.1（幾何掃引の同形状比較 — .claude/reviews/2026-09-29_quant-default-recon §3.2）。
- M2（Chrome / Dawn / Metal）は Tint が `dot4I8Packed` を展開するので整数内積の利得が無く、幾何の直った f32 経路（×1.78）に
  抜かれて既定 quant が 1.58 倍遅い。
- 「i8 が強い」のは重みが i8 だからではなく a8 の整数内積の効き。i8 重み × f32 計算（`f16+dit8`）は DL と重み VRAM を減らす
  だけで、計算時間は f16 と同じはず（M2 / RTX とも未計測 — §7 (b)）。
- B570 の GPU 時間は Deno の raw tick（1 tick = 52.08 ns）を換算した値（§3）。

## 13. gpu-lab の quick+ — Apple M5 対 M2（2026-09-29 夜・時点スナップショット）

利用者の M5（Chrome 153・`apple` / `metal-3`・キーボード面を 12 cm ファンで冷却）と M2 で quick+（45 ケース・340 行・失敗 0・不一致 0・
反復上限 16,384）を回した。記録 = `outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-37-17.817Z.json`（M5）/
`…T21-40-45.831Z.json`（M2）。M5 の既定の再測定比は 44 / 45 ケースが 0.9〜1.1 の内（外は attention ③PV cross M 1024 の 0.890 — 1 dispatch
1.3 ms の短いケース）。冷却無しの走行は記録が無いので比較していない。

| ケース（既定幾何）          | M5 既定 ms | M2 既定 ms | M2 / M5 | `apple-metal-3` の採用幾何の比 M5 / M2 |
| --------------------------- | ---------: | ---------: | ------: | -------------------------------------: |
| linear M 4096 N 2048 K 2048 |      17.79 |      79.07 |    4.45 |       ×0.96 / ×1.75（reg128x32r8x4w8） |
| linear M 1024 N 8192 K 2048 |      17.32 |      78.93 |    4.56 |                  ×0.97 / ×1.75（同上） |
| attention ①QK self M 4096   |      37.76 |     163.29 |    4.32 |                  ×0.96 / ×1.75（同上） |
| attention ③PV self M 4096   |      33.36 |     128.22 |    3.84 |       ×0.96 / ×1.70（reg64x64r8x4w16） |
| conv2d Cout 192 256²        |      25.41 |     103.46 |    4.07 |   ×0.82 / ×1.22（igemm128x64:wg16x16） |
| conv2d Cout 96 512²         |      31.82 |     137.93 |    4.33 |                  ×1.05 / ×1.69（同上） |
| i8a8 linear M 4096          |      22.42 |      80.47 |    3.59 | ×1.05 / ×1.12（tile64x64r8x4w16x8k16） |
| linear M 64 N 3072 K 1024   |       0.54 |       1.41 |    2.60 |       ×2.00 / ×2.25（reg64x64r4x4w16） |
| linear M 512 N 2048 K 1024  |       1.35 |       3.18 |    2.36 |       ×1.10 / ×1.10（reg128x32r8x4w8） |

- M5 は既定幾何どうしで大 GEMM が M2 の 4.3〜4.5 倍速く、小 M では 2.1〜2.6 倍。
- M2 で勝った大タイル→細タイルの置き換え（`reg128x32` など）は M5 では 4〜11% 遅い。M5 の最良はほぼ既定（128×128）で、
  M5 単独の quick+ から作る表は conv2d rows32 の `igemm128x128` ×1.11 以外すべて既定。M5 で ×1.05 以上取れるのは小 M（M 64 ×2.00・
  M 32 ×1.58・bmm ×1.39〜1.46）と conv2d の一部（×1.11〜1.21）。
- Chrome は M5 も `metal-3` と名乗るので、今の登録では M5 に `apple-metal-3` が当たり DiT 段で約 5% 遅くなる（ADR 0115 追記 9・裁定待ち）。
  M5 で既定に戻すには gpu-lab の GPU 設定で `default` を注入する。
- RTX 5070 Ti の Anima（gpu-lab・512²）: `default` 注入と表の注入で PNG sha は f16 / 既定 quant とも一致。既定 quant の DiT 段 GPU
  1.34〜1.38 s → 1.13〜1.16 s（×1.19）。

## 14. gpu-lab の full — Apple M5（2026-09-29 深夜・時点スナップショット）

利用者の M5（Chrome 154・`apple` / `metal-3`・冷却あり〈利用者の申告 — 記録には残らない〉）で full（7 族 45 ケース・2,247 行・失敗 0・
不一致 0・反復上限 16,384・GPU timestamp ns・非量子化）を回した。記録 = `outputs/bench-browser/geometry-sweep-browser-2026-09-29T22-31-21.284Z.json`
（sha256 `3cbda2bbd4c2`…・checkout `575b0b46`・dirty 無し）。既定の再測定比は 42 / 45 ケースが 0.9〜1.1 の内で、外の 3 件は
f32 attention の ③PV self M 1024（1.266）・①QK cross M 1024（1.171）・①QK cross M 4096（1.119）。quick+（§13）は 44 / 45 が内で
（外は ③PV cross M 1024 の 0.890）、範囲外のケースは両者で重ならない。同じケースの既定の 1 dispatch は quick+ と full で 0.92〜1.10 倍の差。

訂正: §13 は M5 の Chrome を 153 と書いたが、quick+ の記録（`…T21-37-17.817Z.json`）の userAgent は Chrome 154（M2 の `…T21-40-45.831Z.json` は 153）。

既定と最良（最良が既定の何倍速いか）:

| 群                                                           | ケース | 既定比の範囲 | 代表（既定幾何 → 最良幾何・比）                                                                                                   |
| ------------------------------------------------------------ | -----: | -----------: | --------------------------------------------------------------------------------------------------------------------------------- |
| f32 GEMM M ≥ 1024（linear 6・matmul M 4096）                 |      7 |  ×1.00〜1.03 | 5 ケースで既定 `reg128x128r8x8w16` が最良・残る linear 2 ケース（K 8192 M 1024・N 8192 M 4096）は `reg128x64r8x8w8` ×1.02 / ×1.03 |
| i8a8 linear                                                  |      6 |  ×1.04〜1.08 | 既定 `tile128x64r8x8w8x16k16` → linear M 4096 N 2048 K 2048 で `tile128x32r8x4w8x16k16` ×1.08                                     |
| f32 attention ①QK / ③PV                                      |      8 |  ×1.00〜1.10 | 既定 `reg128x128r8x8w16` → ①QK self M 1024 で `reg128x64r8x8w8` ×1.10                                                             |
| i8a8 attention ①QK / ③PV                                     |      8 |  ×1.00〜1.11 | ①QK self M 4096 で `tile64x64r8x4w16x8k32` ×1.11                                                                                  |
| 中 M（linear M 128 / 256 / 512・matmul M 512・bmm M 512 ×3） |      7 |  ×1.01〜1.16 | 既定 `reg64x32r4x4w8` → linear M 512 で `reg128x64r8x8w8` ×1.16・matmul M 512 で `reg64x128r8x8w16` ×1.13                         |
| 小 M（linear M 16 / 32 / 64・matmul M 64・bmm M 64 ×2）      |      6 |  ×1.07〜2.06 | 既定 `reg16x16r1x4w4` → linear M 64 `reg64x64r4x8w8` ×2.06・M 32 `reg32x32r4x4w8` ×1.54・matmul M 64 ×1.57・bmm M 64 ×1.35〜1.43  |
| conv2d                                                       |      3 |  ×1.00〜1.17 | Cout 384 128² で `igemm128x64:wg8x16` ×1.17・Cout 96 512² で同 ×1.07・Cout 192 は既定 `igemm64x128:wg16x8`                        |

- 大形状（M ≥ 1024 の GEMM・attention）と i8a8 は既定がほぼ最良（×1.11 以内）。範囲外の 3 件は f32 attention なので、その比は機の揺れを含む。
  差が開くのは小 M（M 64 ×2.06・M 32 ×1.54・matmul M 64 ×1.57・bmm M 64 ×1.35〜1.43）と conv2d Cout 384（×1.17）、次いで中 M 512（×1.13〜1.16）。
- M5 と M2（full 同士・`…T21-04-14.586Z.json`）の既定幾何の 1 dispatch 時間比（M2 ÷ M5）: f32 linear（M ≥ 1024 の 6 形状）×4.26〜4.46・
  i8a8 linear（同 6 形状）×3.59〜3.68・f32 attention ×3.50〜4.37・i8a8 attention ×3.47〜3.69。§13（quick+）の大 GEMM 4.3〜4.5 倍と同じ傾向。

M5 の full 単独から生成器を回した試走（`--min-speedup 1.05`・`--description "Apple M5"`・生成物はリポ外 — 登録しない）:

- 採用は 2 欄だけ: `gemmRows` ≤ 64 = `reg16x32r2x4w8` ×1.185（6 ケースで ×1.054〜1.352）・conv2d rows32 = `igemm128x64:wg8x16` ×1.071（1 ケース）。
  他の 8 欄（65〜512・> 512・f32 attention 2 欄・conv2d rows64・i8a8 3 欄）は既定。範囲外の 3 件は比の材料から外れた（ADR 0115 追記決定 8）。
- ≤ 64 で M 64 の ×2.06 が取れない理由: 欄は 6 ケース全てで ×1.05 以上の幾何しか採らない。M 64 の最良 `reg64x64r4x8w8` は M 16 で ×0.62 に落ち、
  採用の `reg16x32r2x4w8` は M 64 では ×1.13 に留まる。行数バケットの細分化（ADR 0115 追記 9 の裁定 3）の動機を M5 でも裏付ける。
- §13 の「M5 単独の quick+ から作る表は conv2d rows32 以外すべて既定」と結論が変わったのは候補の差: `reg16x32r2x4w8` と `igemm128x64:wg8x16` は
  quick+ の候補に無い。quick+ で rows32 に残った `igemm128x128:wg16x16`（Cout 96 ×1.11）は full では ×1.03 で門に届かない。

`apple-metal-3`（M2 の表）を M5 に当てたときの比（M5 full の行から引いた既定比）:

| 欄                           | HEAD の採用幾何（追記決定 8 で再生成）            |       M5 での既定比 | checkout `575b0b46` の版（M5 の掃引・§13 の当時） |
| ---------------------------- | ------------------------------------------------- | ------------------: | ------------------------------------------------- |
| `gemmRows` ≤ 64 / 65〜512    | 既定と同じ（`reg16x16r1x4w4` / `reg64x32r4x4w8`） |               ×1.00 | 同じ                                              |
| `gemmRows` > 512（7 ケース） | `reg128x32r8x4w8`                                 |         ×0.89〜0.98 | 同じ                                              |
| `attention.qk`               | `reg128x32r8x4w8`                                 |         ×0.92〜1.05 | 同じ                                              |
| `attention.pv`               | `reg32x64r4x8w8`                                  |         ×0.88〜1.03 | `reg64x64r8x4w16` ×0.82〜0.99                     |
| `conv2d` rows64 / rows32     | `igemm128x64:wg16x16`                             | ×0.77〜1.10 / ×0.99 | 同じ                                              |
| `i8a8.linear`                | `tile64x64r4x8w8x16k16`                           |         ×0.97〜1.06 | `tile64x64r8x4w16x8k16` ×1.02〜1.08               |
| `i8a8.attentionQk`           | `tile32x64r4x8w8x8k16`                            |         ×0.98〜1.09 | `tile64x64r8x4w16x8k16` ×1.00〜1.10               |
| `i8a8.attentionPv`           | `tile16x128r4x8w16x4k16`                          |         ×0.94〜1.01 | `tile64x64r8x4w16x8k16` ×0.96〜1.03               |

- 全欄で ×0.77〜1.10。f32 の M2 向け幾何は M5 で多くのケースが負け（既定を超えるのは conv2d Cout 384 ×1.10・①QK cross M 4096 ×1.05・
  ③PV cross M 1024 ×1.03 の 3 件）、i8a8 はほぼ中立。quick+（§13・ADR 0115 追記 9）の観察と同じ結論。
- 今の照合（ADR 0115 追記決定 7）では `apple-metal-3` は adapter の `description` が `"Apple M2"` のときだけ当たり、M5 には当たらないので実害は無い
  （利用者が注入した場合だけこの比になる）。

i8a8 対 f32 計算（同形状 14 本 = linear 6・attention 8。f32 側の linear の重みは f16〈`:wf16`〉）の 1 dispatch 時間比（i8a8 ÷ f32・1 より大きいと i8a8 が遅い）:

| 機（full の記録）                     | 既定同士 linear / attention | 最良同士 linear / attention | 既定の TFLOPS i8a8 / f32（linear） | 既定の TFLOPS i8a8 / f32（attention） |
| ------------------------------------- | --------------------------: | --------------------------: | ---------------------------------: | ------------------------------------: |
| Apple M5（`…T22-31-21.284Z.json`）    |     1.19〜1.28 / 1.18〜1.45 |     1.14〜1.19 / 1.17〜1.51 |            1.51〜1.58 / 1.83〜1.95 |               1.39〜1.51 / 1.67〜2.13 |
| Apple M2（`…T21-04-14.586Z.json`）    |     1.01〜1.06 / 1.00〜1.37 |     1.54〜1.60 / 1.54〜2.00 |            0.41〜0.43 / 0.43〜0.44 |               0.40〜0.41 / 0.40〜0.56 |
| RTX 5070 Ti（`…T20-42-08.545Z.json`） |     0.30〜0.36 / 0.34〜0.52 |     0.27〜0.29 / 0.28〜0.40 |            32.8〜38.7 / 11.6〜13.3 |               25.8〜32.5 / 10.6〜13.3 |

- M5 でも i8a8 は f32 計算より遅い（既定同士 linear 1.19〜1.28 倍・attention 1.18〜1.45 倍・最良同士 1.14〜1.51 倍）。M2 から M5 への伸びは
  f32（linear ×4.26〜4.46）が i8a8（×3.59〜3.68）より大きく、既定同士の差は M2（linear 1.01〜1.06 倍）より開いた。最良同士では
  M2（1.54〜1.60 倍）の方が開いている（M2 は f32 側が幾何で大きく伸びる — §10）。整数内積が効く RTX では i8a8 が f32 の 0.27〜0.52 倍の時間。
- 掃引に f16 演算のケースは無いので、f16 演算との直接比較は無い。M5 の Anima の既定 quant 対 `f16`（DiT 段）は未計測
  （`.claude/reviews/2026-09-29_quant-default-recon/` §5.4「判断に足りない実測」にも M5 の項は無い・git 追跡外）。
- 「M シリーズで整数経路が速くなる時期」は、その前提（int8 が f32 / f16 より速い）が M5 でも成り立たないので調べていない。

## 15. 3 機の full 再走（57 ケース）と 7 段の表・Pixel 10a の quick+（2026-10-01・時点スナップショット）

ADR [0116](../decisions/0116-geometry-profile-row-buckets.md) 段 2 の再走。3 機とも checkout `7be21121`（dirty 無し）・full・全 7 族 57 ケース・2,895 行・
失敗 0・不一致 0・GPU timestamp ns・非量子化・反復上限 16,384。既定の再測定比の範囲外（生成器が比の材料から外す — ADR 0115 追記決定 8）は
M5 の 2 件（matmul M 16 N 3072 K 1024 の 1.169・conv2d Cout 192 256² の 1.111）だけ。

| 機          | 記録（`outputs/bench-browser/geometry-sweep-browser-…`） | sha256（先頭） | Chrome |
| ----------- | -------------------------------------------------------- | -------------- | ------ |
| Apple M2    | `…2026-10-01T12-28-47.052Z.json`                         | `11fed6a8`     | 154    |
| Apple M5    | `…2026-10-01T16-28-06.082Z.json`                         | `dac3e6be`     | 154    |
| RTX 5070 Ti | `…2026-10-01T16-46-50.379Z.json`                         | `0e57d4ef`     | 154    |

7 段の `gemmRows` の採否（生成器 `--min-speedup 1.05`・値は「採用幾何 幾何平均（最小〜最大）」・「既定」は採用なし）。M2 は `apple-metal-3` の
材料 5 本（既存 4 本 + 再走）、RTX は `nvidia-blackwell` の材料 3 本（既存 2 本 + 再走）で、どちらも登録済み（`c358832b`）。M5 は 2 本
（§14 の full + 再走）からの試走で登録しない。括弧内は ADR 0116 Context のシミュレーション（45 ケース・linear だけの段あり）の予告:

| 段       | ケース                      | M2（5 本）                                                             | M5（2 本・試走）                                                      | RTX（3 本）                                                 |
| -------- | --------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------- |
| ≤ 16     | linear 1 + matmul 1 + bmm 2 | 既定（予告 `reg16x32r2x4w8` ×1.209）                                   | 既定（予告 `reg16x32r2x4w8` ×1.065）                                  | 既定（予告どおり）                                          |
| 17〜32   | linear 1 + matmul 1 + bmm 2 | `reg32x32r2x4w8` ×1.308（1.153〜1.633・予告 `reg32x64r4x4w16` ×1.750） | `reg32x32r2x4w8` ×1.246（1.058〜1.461・予告 `reg32x32r4x4w8` ×1.541） | 既定（予告 `reg32x16r2x4w4` ×1.072）                        |
| 33〜64   | linear 1 + matmul 1 + bmm 2 | `reg64x32r4x4w8` ×1.412（1.142〜2.236・予告どおり ×1.397）             | `reg64x32r4x4w8` ×1.538（1.290〜1.938・予告どおり ×1.514）            | 既定（予告どおり）                                          |
| 65〜128  | linear 1 + matmul 1 + bmm 2 | 既定（予告 `reg128x32r8x4w8` ×1.100）                                  | 既定（予告どおり）                                                    | 既定（予告どおり）                                          |
| 129〜256 | linear 1 + matmul 1 + bmm 2 | 既定（予告 `reg128x32r8x4w8` ×1.054）                                  | 既定（予告 `reg128x32r8x4w8` ×1.112）                                 | `reg64x64r4x4w16` ×1.203（1.149〜1.291・予告どおり ×1.215） |
| 257〜512 | linear 1 + matmul 1 + bmm 3 | 既定（予告どおり）                                                     | 既定（予告どおり）                                                    | `reg64x64r8x4w16` ×1.330（1.184〜1.451・予告どおり ×1.338） |
| > 512    | linear 6 + matmul 1         | `reg128x32r8x4w8` ×1.513（1.145〜1.600・予告どおり ×1.647）            | 既定（予告どおり）                                                    | 既定（予告どおり）                                          |

- linear だけで決めていた 4 段（≤ 16 / 17〜32 / 65〜128 / 129〜256）× 3 機 = 12 段のうち、予告の採用は 9 段、matmul / bmm のケースが入った
  実結果の採用は 3 段。6 段は既定へ戻り、2 段（M2 / M5 の 17〜32）は幾何が変わり、1 段（RTX 129〜256）だけ予告どおり。×1.05 ぎりぎりの予告
  3 件（M5 ≤ 16・M2 129〜256・RTX 17〜32）はすべて既定へ戻った。3 経路を持っていた段（33〜64 / 257〜512 / > 512）は 9 段すべて予告どおり。
- `gemmRows` 以外の欄: M2 は attention ③PV が `reg32x64r4x8w8` ×1.635 → `reg64x64r4x8w8` ×1.661（再走の観測が幾何平均を動かした）で、
  ①QK `reg128x32r8x4w8` ×1.527・conv2d `igemm128x64:wg16x16` ×1.398 / ×1.600・i8a8 3 欄（×1.116 / ×1.157 / ×1.131）は幾何不変。RTX は f32 の
  attention / conv2d が既定のまま、i8a8 3 欄 `tile128x64r8x4w16x16k16`（×1.194 / ×1.218 / ×1.210）も不変。M5 の試走は上の 2 段と
  conv2d rows32 `igemm32x64:wg8x8` ×1.097 以外すべて既定（§14 の「既定がほぼ最良」のまま）。
- 既定の linear M 1024 N 2048 K 2048 の 1 dispatch: M2 13.07 ms・M5 4.41 ms・RTX 0.70 ms（§14 の M2 ÷ M5 ≈ 4.3 倍と同じ・RTX は M5 の 6.3 倍）。

### Pixel 10a（Android Chrome 154・adapter `arm` / `valhall` / `Mali-G715`）— quick+ と Anima（利用者の実走・旧 checkout）

利用者が devtools のポート転送越しに gpu-lab を回した記録。checkout `76812e94`（`7be21121` の 10 コミット前 — 新ケース 12 本の追加前で 45 ケース・
生成器は 3 段）なので、本節の数字は旧版の時点スナップショット。再走は未。

- 掃引 `outputs/bench-browser/geometry-sweep-browser-2026-10-01T09-37-40.824Z.json`（sha256 `14cb9120`…・quick+・385 行・失敗 0・不一致 0・
  再測定比の範囲外 0・GPU timestamp ns・非量子化・dp4a あり）。既定の linear M 1024 N 2048 K 2048 は 1 dispatch 89.8 ms（M5 の 20 倍・RTX の 128 倍）。
- HEAD の生成器（7 段・`--min-speedup 1.05`）を quick+ の記録 1 本に掛けた試走（登録しない）: 14 欄中 13 欄で採用・既定比 ×1.25〜2.83。
  `gemmRows` は ≤ 16 だけ既定で、17〜32 `reg32x64r4x8w8` ×1.69・33〜64 `reg64x64r8x4w16` ×1.95（1.34〜2.71）・65〜128 `reg64x64r4x4w16` ×1.25・
  129〜256 `reg64x64r8x4w16` ×1.31・257〜512 同 ×2.14（1.42〜2.52）・> 512 同 ×2.21（2.00〜2.87）。①QK 同 ×2.38・③PV 同 ×2.70・
  conv2d `igemm64x64:wg16x8` ×2.28 / `igemm128x64:wg16x16` ×2.49・i8a8 3 欄 `tile128x64r8x4w16x16k16` ×2.83 / ×2.76 / ×2.69。
  M 16 / 32 / 128 / 256 の段は linear 1 ケースだけ（旧ケース集合）なので、§「linear しか無い段」の偏りをそのまま含む。
- Anima 512²（seed 42・既定の prompt・常駐 transformer）。表 = 端末上の gpu-lab が 3 段で生成した `generated:arm-valhall`
  （≤ 64 既定・65〜512 と > 512 が `reg64x64r8x4w16`）。記録 `anima-residency-browser-f16-2026-10-01T10-23-19.650Z.json`（表を注入）と
  `…T10-43-00.425Z.json`（`auto` = 既定）:

| quant                           | 表   |    全体 | text_encoder | text_conditioner |                                                          transformer（DiT） | vae_decoder | PNG sha256（先頭） |
| ------------------------------- | ---- | ------: | -----------: | ---------------: | --------------------------------------------------------------------------: | ----------: | ------------------ |
| 既定（`f16+dit8-a8-attn8-s16`） | 注入 | 234.3 s |       50.6 s |           13.7 s |                                                                     156.1 s |      13.9 s | `e5f1a49d`         |
| 既定（`f16+dit8-a8-attn8-s16`） | 既定 | 318.7 s |       49.8 s |           13.9 s |                                                                     234.3 s |      20.3 s | `e5f1a49d`         |
| `f16`                           | 注入 | 389.3 s |       50.1 s |           13.6 s |                                                                     311.7 s |      13.9 s | `bdccacc7`         |
| `f16`                           | 既定 |    失敗 |       50.3 s |           13.7 s | 471 s で device 消失（`OperationError: Instance dropped in popErrorScope`） |           — | —                  |

- 表で DiT 段 ×1.50（234.3 → 156.1 s）・VAE ×1.46・全体 ×1.36。PNG は表の有無で一致（幾何は担当割りだけ、が Mali でも成り立つ）。
  M2 / RTX の PNG（§11 / §12）とは一致しない（機が違えば f32 の丸めが違う — 機を跨ぐ一致は主張していない）。
- Mali では既定 quant（a8）が `f16` より速い（DiT 156 s 対 312 s = 2.0 倍）。Apple（§12・§14: a8 が遅い）とは逆で、B570 / RTX と同じ側。
- text_encoder の約 50 s は 4 走とも同じで、ポート転送越しの重み取得（§「text 44 s」と同じ帰属）。
- `f16` の既定幾何の走行は DiT の途中で device が落ちた（推測: 8 GB 級 Android の VRAM 圧 — known-issues「Pixel（8GB 級 Android Chrome）」と同じ系統）。表ありの `f16` は完走した。

### M2 / RTX 5070 Ti の自己 A/B — 7 段の表（既定 quant・2026-10-01・checkout `c017044a`）

gpu-lab の Anima タブの「A/B（既定 vs 表）」で、既定 quant（`f16+dit8-a8-attn8-s16`）・512²・seed 42・N = 3 を、`default` を注入した
区間 A と `自動`（M2 = `apple-metal-3`・RTX = `nvidia-blackwell`）の区間 B で回した記録（`outputs/bench-browser/anima-residency-browser-f16+dit8-a8-attn8-s16-2026-10-01T18-47-27.991Z.json` = M2・
`…T18-29-32.207Z.json` = RTX）。時間は各区間の 2 回目以降（常駐 2 回目以降）。

| 機          | PNG sha256（先頭）           | DiT 段 壁時計 A → B     | DiT 段 GPU A → B             | VAE 段 GPU A → B | text_encoder 壁時計    |
| ----------- | ---------------------------- | ----------------------- | ---------------------------- | ---------------- | ---------------------- |
| Apple M2    | `0a5695470e4a`（6 / 6 一致） | 78.1 → 72.5 s（×0.928） | 76.85 → 71.08 s（×0.925）    | 2.92 → 2.40 s    | 45〜48 s（ポート転送） |
| RTX 5070 Ti | `3b07b912c4d4`（6 / 6 一致） | 1.40 → 1.22 s（×0.87）  | 1.32 → 1.12〜1.17 s（×0.87） | 0.12 → 0.12 s    | 12.8 s                 |

- PNG は 2 機とも区間内・区間間で一致し、2026-09-29 の記録（§11 / §12）の既定 quant の sha とも同じ — 7 段の表でもビット同一は保たれた。
- 既定 quant の DiT は i8a8 の linear / attention で走るので、この席で効くのは i8a8 の 3 欄（7 段化で不変）。M2 の ③PV の変更
  （`reg64x64r4x8w8`）と RTX の M 129〜512 の 2 段は f32 経路（`f16` 席の DiT・text 段の k / v 射影）に効くので、**`f16` 席の A/B が
  7 段の変更の検収**になる（未取得 — ADR 0116 検収 段 5 の残り → M2 は次の小節〈2026-10-02〉で取得）。既定 quant の数字は「変更前と同じ比が出ている・退行なし」の確認。
- text_encoder の 45 s（M2）/ 12.8 s（RTX）は取得経路（ポート転送 / ネットワーク）の時間で、GPU 時間は 0.04〜0.13 s。

### M2 の自己 A/B — `f16` 席（2026-10-02・checkout `bbcbcc49`・Chrome 154）

gpu-lab の Anima タブの A/B ボタン（既定 quant と `f16` の 2 席固定）で、512²・seed 42・N = 3・DiT 常駐を、`default` を注入した
区間 A と `自動`（= `apple-metal-3`）の区間 B で回した記録（`outputs/bench-browser/anima-residency-browser-f16+dit8-a8-attn8-s16-2026-10-02T04-26-30.527Z.json`・
2026-10-02T04:26Z・adapter `apple` / `metal-3` / `"Apple M2"`・12 行・失敗 0・checkoutDirty false）。時間は各区間の 2 回目以降の中央値。
各段の geometryProfile は区間 A が `default`・区間 B が `apple-metal-3`。

`f16` 席:

| 項目                       | 区間 A（`default`）            | 区間 B（`apple-metal-3`）    | B / A        |
| -------------------------- | ------------------------------ | ---------------------------- | ------------ |
| PNG sha256（先頭）         | `041027e63559`（3 / 3）        | `041027e63559`（3 / 3）      | 6 / 6 一致   |
| 全体（壁時計）             | 119.2 s                        | 111.8 s                      | ×0.938       |
| DiT 段（壁時計）           | 52.0 s                         | 43.3 s                       | ×0.833       |
| DiT 段（GPU）              | 50.75 s                        | 42.02 s                      | ×0.828       |
| 内訳 linear（f32）         | 41.80 s（`reg128x128r8x8w16`） | 34.52 s（`reg128x32r8x4w8`） | ×0.826       |
| 内訳 ①QK                   | 2.32 s（`reg128x128r8x8w16`）  | 1.97 s（`reg128x32r8x4w8`）  | ×0.85        |
| 内訳 ③PV                   | 2.64 s（`reg128x128r8x8w16`）  | 1.53 s（`reg64x64r4x8w8`）   | ×0.58        |
| VAE 段（GPU / 壁時計）     | 2.92 s / 9.5 s                 | 2.30 s / 9.2 s               | ×0.79（GPU） |
| text_conditioner 段（GPU） | 0.041 s                        | 0.032 s                      | —            |
| text_encoder 段（GPU）     | 0.126 s                        | 0.091 s                      | —            |

- text_conditioner の壁時計は 12.1 s で両区間同じ（ホスト側の時間）。text_encoder の壁時計 45〜47 s はポート転送越しの重み取得。
- 同じ走行の既定 quant（`f16+dit8-a8-attn8-s16`）: PNG `0a5695470e4a` 6 / 6 一致・全体 145.3 → 135.6 s（×0.933）・DiT 段 壁時計
  78.1 → 69.6 s（×0.890）・GPU 76.85 → 68.42 s（×0.890）・VAE 段 GPU 2.91 → 2.31 s。sha と区間 A の DiT（78.1 s / 76.85 s）は
  2026-10-01 の記録（上の小節）と同じ。

既定側が 2026-09-27 より速い。09-29（§11・Chrome 153）の表側と今日の表側はほぼ同じで、違うのは既定側:

| 側                      | 2026-09-29 / 09-27（Chrome 153）                                    | 2026-10-02（Chrome 154）                                             |
| ----------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 表側（`apple-metal-3`） | DiT GPU 44.2 s（linear 37.1・①QK 1.99・③PV `reg64x64r8x4w16` 1.61） | DiT GPU 42.02 s（linear 34.52・①QK 1.97・③PV `reg64x64r4x8w8` 1.53） |
| 既定側（`default`）     | DiT GPU 77.3 s（linear 65.4 s・09-27）                              | DiT GPU 50.75 s（linear 41.80 s）                                    |

09-29 の値は `anima-residency-browser-f16-2026-09-29T17-15-47.294Z.json`（checkout `cb87ab93`）。同じケース・同じ WGSL で掃引記録を
突合した（M2 full・09-29 = `geometry-sweep-browser-2026-09-29T21-04-14.586Z.json`〈Chrome 153・checkout `575b0b46`〉・10-01 =
`geometry-sweep-browser-2026-10-01T12-28-47.052Z.json`〈Chrome 154・checkout `7be21121`〉）。2 つの checkout の間の `packages/runtime/src` の
変更は、表（`kernels/geometry-profiles`）と表の選択・凍結の周辺（`geometry-profile.ts` の match に description を足す・
`index.ts` の一覧の凍結・コメント）だけで、カーネルの codegen は不変 — 既定行のキーと outputSha256 も 2 本で同一なので、既定幾何の
WGSL は同一:

| ケース                        | 既定 `reg128x128r8x8w16` 09-29 → 10-01 | 今の表の幾何 09-29 → 10-01         | 表 / 既定の速さ |
| ----------------------------- | -------------------------------------- | ---------------------------------- | --------------- |
| linear-m1024-n2048-k2048      | 19.68 → 13.07 ms（×0.66）              | `reg128x32r8x4w8` 11.38 → 11.00 ms | ×1.73 → ×1.19   |
| attention-qk-self-m1024-n1024 | 10.59 → 6.99 ms                        | `reg128x32r8x4w8` 5.86 → 5.84 ms   | ×1.81 → ×1.20   |
| attention-qk-self-m4096-n4096 | 162.6 → 104.4 ms                       | `reg128x32r8x4w8` 93.3 → 92.8 ms   | —               |
| attention-pv-self-m1024-n1024 | 7.90 → 7.84 ms（不変）                 | `reg64x64r4x8w8` 4.76 → 4.53 ms    | —               |
| attention-pv-self-m4096-n4096 | 125.9 → 128.4 ms                       | `reg64x64r4x8w8` 75.4 → 71.7 ms    | —               |

- 事実: 既定 `reg128x128r8x8w16` の linear / ①QK が縮み（全既定行で linear ×0.65〜0.66・①QK ×0.64〜0.66）、③PV の既定カーネルと
  matmul M 4096 の既定は不変（×0.99〜1.02）。表の幾何は linear / ①QK で不変、③PV の `reg64x64r4x8w8` は約 5% 速い。既定の
  conv2d（igemm）も ×0.83〜0.85、中くらいの M の linear `reg64x32r4x4w8` も ×0.87〜0.94 に縮んでおり、VAE 段の既定 GPU
  （09-27 の 3.9〜4.4 s → 今日 2.92 s）に効いている。
- 帰属（推定）: 同じ機・同じ WGSL で記録から読める差は Chrome 153 → 154 だけ。UA の OS 表記は 10_15_7 で変わらず、同時期の
  OS / Metal ドライバの更新の有無は記録から読めないので、「Chrome 153 → 154（または同時期の OS / Metal ドライバ更新）」までしか
  言えない。

含意:

1. 7 段の表の `f16` 席の DiT の利得は in situ で ×1.20（壁時計 52.0 → 43.3 s・GPU ×1.21）。§11 の ×1.78（80.5 → 45.1 s）は Chrome 153 の
   既定との比。③PV は A/B 内で 2.64 → 1.53 s（×0.58）。09-29 の旧幾何 `reg64x64r8x4w16` 1.61 s との差は Chrome 版をまたぐので、
   幾何の変更だけの効果とは切り分けられない（掃引では Chrome 153 で旧新同速〈4.765 / 4.759 ms〉・Chrome 154 で新が 8% 速い
   〈4.949 / 4.533 ms〉）。PNG sha は 6 / 6 一致で退行なし。
2. perf-ledger K-71 の kill 線「M2 の DiT 段が `f16` quant で 1.3 倍未満なら Apple プロファイルを再検討」を比では下回る。絶対値では
   表が最速（43.3 s 対 52.0 s）。利用者裁定待ち。
3. `apple-metal-3`（`c358832b`）の材料 5 本のうち 4 本（10-01 より前の M2 の掃引）は Chrome 153、1 本（10-01 の再走）が Chrome 154。表に
   書かれた比のうち、既定が速くなった欄（> 512 `reg128x32r8x4w8` ×1.51・①QK）は Chrome 154 の実態（10-01 単独の幾何平均で > 512
   ×1.18・①QK ×1.16）より大きい。③PV は既定が速くなっていないので逆で、表の ×1.66 より 10-01 単独の ×1.75 のほうが大きい。幾何の
   選択（どの幾何が最速か）は 3 欄とも変わらない。ADR [0117](../decisions/0117-app-geometry-tuning.md) §10「検出できない古さ」（ブラウザ・
   ドライバ・OS の更新で性能特性が変わっても照合は一致のまま）の最初の実例。

### RTX 5070 Ti の自己 A/B — `f16` 席（2026-10-02・checkout `bb6d055e`・Chrome 154）

M2 と同じ手順（gpu-lab の A/B ボタン・既定 quant と `f16` の 2 席・512²・seed 42・N = 3・区間 A = `default` 注入・区間 B = `自動` =
`nvidia-blackwell`）の記録（`outputs/bench-browser/anima-residency-browser-f16+dit8-a8-attn8-s16-2026-10-02T05-46-09.428Z.json`・
adapter `nvidia` / `blackwell` / `0x2c05` / `"NVIDIA GeForce RTX 5070 Ti"`・12 行・失敗 0・checkoutDirty true = ADR 0117 段 4 の
作業中の未コミット変更〈provenance / tune — codegen には触れない〉が bundle に乗っていた）。時間は各区間の 2 回目以降の中央値。

| 席                | PNG sha256（先頭）           | DiT 段 GPU A → B          | 全体 壁時計 A → B         | 備考                                                                                    |
| ----------------- | ---------------------------- | ------------------------- | ------------------------- | --------------------------------------------------------------------------------------- |
| 既定（i8a8・s16） | `3b07b912c4d4`（6 / 6 一致） | 1.346 → 1.178 s（×0.875） | 18.19 → 18.06 s（×0.993） | 10-01 の記録（1.32 → 1.12〜1.17・×0.87）と同じ比・同じ sha                              |
| `f16`             | `c3cef8d6bc64`（6 / 6 一致） | 2.876 → 2.875 s（×1.000） | 19.74 → 19.76 s（×1.001） | DiT の f32 linear（M > 512）は `nvidia-blackwell` では既定のまま — 変化なしが期待どおり |

- `nvidia-blackwell` の 7 段化で新しく入った 257〜512 の規則（`reg64x64r8x4w16`）は、`f16` 席の DiT の中では M = 512 の linear
  （cross-attention の k / v 射影）に当たり、その行は `reg64x32r4x4w8` 0.139 s → `reg64x64r8x4w16` 0.106 s（×0.76・掃引の ×1.33 と
  同じ向き）。DiT 全体では他の行の揺れに埋もれて中立（2.876 → 2.875 s）。text_conditioner 段の GPU は 0.011 s で両区間同じ。
- text_encoder の壁時計 12.8 s はネットワーク越しの重み取得（GPU 0.03〜0.04 s）。VAE 段 GPU 0.123 s は両区間同じ。
- これで ADR 0116 検収 段 5（M2 / RTX の `f16` 席の自己 A/B・PNG sha 一致・text 段の GPU 時間の記録）は両機とも取得済み。

## 参照

- M2 の JSON: `outputs/bench-browser/anima-residency-browser-f16+dit8-a8-attn8-s16-2026-09-27T14-24-06.310Z.json`
  （format `karume-anima-residency-browser/2`・8 行）
- B570 の JSON: `outputs/bench/karume/2026-09-27_metal-recon/anima-profile-intel-r-graphics-bmg-g21-f16-512x512-2026-09-27T12-59-55.939Z.json`
  と `…-f16+dit8-a8-attn8-s16-512x512-2026-09-27T13-00-17.267Z.json`
- 先行の M2 実測: `outputs/bench-browser/anima-residency-browser-2026-09-27T12-19-09.299Z.json`（[research 2026-09-27 H-35](2026-09-27-h35-oom-device-lost.md) §6）
- 幾何掃引の JSON（§9 / §10）: M2 quick `outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json`・M2 full `…-2026-09-27T18-31-46.471Z.json`・RTX 5070 Ti full `…-2026-09-27T18-37-27.914Z.json`
- Codex の棚卸し（読み取り調査・仮説 3 本）: `.claude/reviews/2026-09-27_metal-recon/codex-inventory.md`（git 追跡外）
- 仕事量の再集計: `.claude/reviews/2026-09-25_perf-recon/deep/B4-hoist-cfg-h30.md` §2.4（git 追跡外）
- 確認ページと双子 CLI の使い方: `tools/anima-residency/browser/README.md`「Per-op GPU timing (K-70)」
