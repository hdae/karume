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

## 参照

- M2 の JSON: `outputs/bench-browser/anima-residency-browser-f16+dit8-a8-attn8-s16-2026-09-27T14-24-06.310Z.json`
  （format `karume-anima-residency-browser/2`・8 行）
- B570 の JSON: `outputs/bench/karume/2026-09-27_metal-recon/anima-profile-intel-r-graphics-bmg-g21-f16-512x512-2026-09-27T12-59-55.939Z.json`
  と `…-f16+dit8-a8-attn8-s16-512x512-2026-09-27T13-00-17.267Z.json`
- 先行の M2 実測: `outputs/bench-browser/anima-residency-browser-2026-09-27T12-19-09.299Z.json`（[research 2026-09-27 H-35](2026-09-27-h35-oom-device-lost.md) §6）
- Codex の棚卸し（読み取り調査・仮説 3 本）: `.claude/reviews/2026-09-27_metal-recon/codex-inventory.md`（git 追跡外）
- 仕事量の再集計: `.claude/reviews/2026-09-25_perf-recon/deep/B4-hoist-cfg-h30.md` §2.4（git 追跡外）
- 確認ページと双子 CLI の使い方: `tools/anima-residency/browser/README.md`「Per-op GPU timing (K-70)」
