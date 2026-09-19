> 2026-09-19 時点（`main` `d1a4377`）の gemma4-qat 実装検証と公式実装突合のスナップショット。読み取り専用のレビューで得た実測で、以後の変更を反映しない。

# QAT レビューの実測記録

裁定と作業項目の正本は [ADR 0097 追記 7](../decisions/0097-gemma4-qat-integration.md)・
[limitations](../limitations.md) の QAT 節・[backlog](../backlog.md)・[perf-ledger](../perf-ledger.md)。
この記録は、その裁定の根拠になった数値と照合結果だけを残す。
実行に使ったスクリプトはレビューの作業領域（`scratchpad/qat-review/Q2`・`Q3`・`Q5`。git 追跡外・保持しない）に置いた。

## 1. SRQ の意味論は実モデルの全 scale で一致した

配布 IR（E2B + E4B）から `static_quantize` の scale を全部抜くと **826 種**。
そのすべてについて、境界 128 本の両隣・厳密な境界値・`±0`・`±Inf`・最小非正規化数・乱数を入力に:

| 比較                                                                             |    入力数 | 結果               |
| -------------------------------------------------------------------------------- | --------: | ------------------ |
| GPU の境界表アルゴリズム（WGSL の逐語移植）と CPU 参照 `referenceStaticQuantize` | 1,062,670 | 相違 0             |
| CPU 参照と公式 `apply_srq`（f32）                                                |   902,550 | ビット単位で相違 0 |

`scale = 0` は NaN payload の比較が harness 側で壊れるため上の掃引から除き、別に恒等であることを確認した。
committed fixture は 30 scale / 24,416 入力なので、実モデルが使う scale 全域を覆う検査は今回が初めて。
常設の門にするなら 826 scale の境界表を現行 fixture と同じ形式で焼くのが素直（容量は数 MB）。
スクリプト: `Q2/srq_sweep.ts`・`Q2/srq_dump.ts`・`Q2/srq_official.py`。

SRQ ノードの本数も合う。E2B は量子化 linear 276 本に対し IR の `static_quantize` は 487 本で、
差の 65 本は「同じ入力・同じ scale」の共通部分式が 1 本に畳まれたぶん
（gate/up の共有 35 層 × 1 と q/k/v の共有 15 層 × 2 = 65）。E4B は 596 本。
checkpoint 側で当該 scale が全層で厳密に等しいことを確認したので、この畳み込みは意味論的に安全。

## 2. 重みの移植は全数照合で未反映 70 本のみ

公式 text の **1,601 テンソル**を全数照合した結果、IR / sidecar に反映されていないのは
`k_cache_scale` / `v_cache_scale` の **70 本だけ**で、これは上流 transformers も読まない（意図的）。
`layer_scalar` 35 本（実値 0.027〜0.44 程度で 1.0 ではない）・`√1536` / `1/√1536`・
`final_logit_softcapping` 30.0 はいずれも IR に載っている。
KV 共有層（層 15〜34）の `k_proj` / `v_proj` は checkpoint に在るが trace に現れないので正しく落ちている。
ビット割り当ても公式 `module_quant_configs` と一致した（head / embed = INT2、PLE は E2B INT4 / E4B INT2、
attention INT4、MLP は層 0-14 が INT4・15 以降が INT2〈E2B〉、per_layer 系 INT8、
`per_layer_model_projection` は非量子化）。
`i4` の group scale は実資産でも行内で完全に定数（列数 > 1 の 110 本を検査・非定数 0 本）なので、
公式の per-row scale `[N,1]` と数値的に同値。

## 3. QAT RoPE の逆周波数が層種別ごとに 1 本だけ公式と 1 ULP 違う

| 層種別                 | karume                 | 公式（transformers）  |
| ---------------------- | ---------------------- | --------------------- |
| sliding `invFreq[111]` | 0.00033982080640271306 | 0.0003398208355065435 |
| full `invFreq[29]`     | 0.20907999575138092    | 0.20908001065254211   |

角度は位置 × 逆周波数なので差は位置に比例し、位置 131,071 の full sin で最大絶対差 **1.909e-3**。
公式の逆周波数を入力に差し替えると、位置 0 / 127 / 4,096 / 131,071 の全点で最大絶対差が
5.96e-8（f32 の 1 ULP）まで落ちる。

出どころは **torch のべき乗が要素数 16 以上でベクトル化経路に切り替わり 1 ULP ずれること**で、
`10000.0` のテンソル冪を長さ 1 / 2 / 4 / 8 で評価すると karume と同値、長さ 16 以上で
2942.72705078125 にずれる（f64 の厳密値 2942.727176209282 の正しい丸めは 2942.727294921875）。
つまり **べき乗段については karume 側が正しい丸め**を得ている。

一方、逆周波数そのものは karume 側も厳密値から 1 ULP 外れる。
full `invFreq[29]` の厳密値は 0.20908000412787182 で、その正しい f32 丸めは公式の
0.20908001065254211 — karume は「べき乗を f32 へ丸めてから逆数を取る」二段丸めのぶん 1 ULP 下になる。
**「karume 側が数学的に正しい」とは書けない**（レビュー中に発見者の当初主張を反証で訂正した）。
単段化（`fround(pow(theta, -(2i)/D))`）に直すと sliding 42/128・full 20/64 が新たに公式と食い違うので、
ADR 0097 追記 5 の段ごと f32 丸めを維持するのが正しい。ビット一致を取りに行くなら、
`pipelineConfig.rope` に逆周波数表（1.5 KB）を焼く案があるが配布資産の再生成が要る。
スクリプト: `Q2/rope_karume.ts`・`Q2/isolate.ts`・`Q2/pow2.py`。

## 4. 「公式 CPU 参照」は 2 種類あり性格が違う

どちらも **transformers（f32・`attn_implementation='eager'`）** であって、公式が想定する
mobile ランタイム（LiteRT-LM）ではない。mobile ランタイムとの一致は測定していない。

| 参照                                         | RoPE                                    | 使いみち                                                                |
| -------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------- |
| 中間値 fixture（`e2b-gpu-cpu-reference.py`） | karume 側の候補 f32 RoPE を注入している | CPU / GPU の中間値がどこで割れたかの帰属。RoPE を揃えた上での縮約差の話 |
| 8 ケース生成（`qat-cpu-broad-reference.py`） | 公式のまま（差し替えなし）              | 生成列の比較。§3 の 1 ULP 差が**含まれている**                          |

## 5. f32 ロードは公式推奨 `dtype="auto"` と同値

| 観点                                             | 実測結果                                                                                                                                        |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| モデルカード推奨 `dtype="auto"` が解決する dtype | **torch.float32**（root config に `dtype` 欄が無く、state_dict の最初の浮動小数テンソル `lm_head.input_activation_scale`〈F32〉から推定される） |
| `GemmaQuantizer` が dtype を上書きするか         | しない（`update_dtype` を override していない）                                                                                                 |
| karume recipe のロード dtype                     | `torch.float32`                                                                                                                                 |

当初「公式は bf16 なのに karume は f32」と疑ったが実測で棄却した。
BF16 の norm 重みはロード時に f32 へ widen されるだけで値は変わらない。

## 6. transformers 5.10.0.dev0 → 5.14.1 に意味論の差は無い

| ファイル                                | 5.10 以降の commit   | 影響                                                                                                                                                     |
| --------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `integrations/gemma_quant.py`           | 1 件（新規追加のみ） | apply_srq / unpack / embedding scale の意味論は一度も変わっていない                                                                                      |
| `quantizers/quantizer_gemma.py`         | 1 件（同上）         | 同上                                                                                                                                                     |
| `models/gemma4/modeling_gemma4.py`      | 9 件                 | text 生成に効く変更は見当たらない（MoE / vision rotary / audio dtype / FSDP / warning 削除 / config 表現）                                               |
| `models/gemma4/configuration_gemma4.py` | 2 件                 | `sliding_window` 半減バグの修正は 5.15 以降。5.14.1 は未修正だが、発火条件 `use_bidirectional_attention == "all"` を本 checkpoint は満たさない（`None`） |

出典: GitHub API `repos/huggingface/transformers/commits?path=…`（本レビュー内で実行）。
pin は `tools/export-recipes/pyproject.toml` の `transformers==5.14.1`。

## 7. 容量ノブの事実と KV メモリ

門の連鎖を CPU だけで実走して確認した（配布 `karume.json` の実値に対し `parseGemma4PipelineConfig` +
`assertChunkLength` を実行）: capacity 128 / 4,096 / 131,072 が OK・31 と 131,073 が拒否、
chunkLength 2 / 32 / 128 が OK・129 と 768 が拒否（**当時の配布宣言 `maxChunkLength = 128` の下での結果**）。

| ノブ                | 硬い上限                    | 上限の由来                                                   |
| ------------------- | --------------------------- | ------------------------------------------------------------ |
| `capacity`          | `maxPosition` = 131,072     | 上流 `config.json` の `max_position_embeddings`              |
| `chunkLength`       | 配布宣言の `maxChunkLength` | trace の記号 `M` の宣言範囲（`trace.py` の `Dim("M", …)`）   |
| `maxNewTokens`      | API 側に上限なし            | 実効上限は `capacity − used − prompt + 1` の 1 式だけ        |
| sliding ring の行数 | 520（窓 512 + slack 8）     | IR に焼いた定数。QAT は投機を使わないので slack は実質未使用 |
| 出口の行数 `R`      | 9                           | `trace.py` の `Dim("R", max=9)`。通常経路は常に 1            |

KV メモリ（f32・実測ではなく IR のスロット実寸からの計算）。
E2B は sliding 12 層が 12.19 MiB の定数 + full 3 層が 12,288 B × capacity、
E4B は 40.62 MiB の定数 + 32,768 B × capacity。

| capacity |     E2B 合計 |     E4B 合計 |
| -------: | -----------: | -----------: |
|      128 |    13.69 MiB |    44.62 MiB |
|      512 |    18.19 MiB |    56.62 MiB |
|    1,024 |    24.19 MiB |    72.62 MiB |
|    4,096 |    60.19 MiB |   168.62 MiB |
|    8,192 |   108.19 MiB |   296.62 MiB |
|   32,768 |   396.19 MiB | 1,064.62 MiB |
|  131,072 | 1,548.19 MiB | 4,136.62 MiB |

スロット 1 本の実寸は E2B が `2048 × C`、E4B が `4096 × C` で、C = 131,072 でそれぞれ 256 MiB / 512 MiB。
manifest の `requiredLimits`（E2B `maxStorageBufferBindingSize: 268435456`、E4B は `maxBufferSize` と
合わせて `536870912`）と厳密に一致するので、宣言を満たす device なら generation context の門には当たらない。
`requiredLimits` は最初から `maxPosition` で焼かれているため、既定 capacity を上げても device 要求は 1 bit も変わらない。

長文脈の正しさに効く要素は個別に照合した: RoPE の長さ依存スケーリングは無い
（`rope_type: "proportional"` は `factor` 既定 1.0 で seq_len を読まない）。
full 層 proportional RoPE の実装は一致（rotary 64 角 + 残り 192 本は invFreq = 0）。
sliding の mask 境界（causal かつ `limit − col < 512`）も、KV 共有層の位置対応も、停止 token `[1, 106, 50]` も一致。

## 8. 公式 mobile との差（karume に無いもの）

| 要素            | 公式 mobile（LiteRT-LM）                                             | karume                                                                                                                         | 影響                                                                              |
| --------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| 活性の実行形    | SRQ で丸めた int8 を**整数内積**（int32 累算・XNNPACK qs8 delegate） | f32 に復元して float で縮約（transformers と同じ）                                                                             | 縮約差が SRQ の丸め境界をまたぎ CPU / GPU で token 列が分岐する根本。速度にも効く |
| KV cache        | 層ごと `k/v_cache_scale` で int8 格納                                | f32（上流 transformers も scale を無視する）                                                                                   | 同じ文脈長でメモリ 4 倍（E2B 32k で 396 MiB vs 99 MiB）・decode の帯域も 4 倍     |
| thinking モード | `enable_thinking` で `<\|think\|>` を描画                            | 非対応（`reasoning` 欄は fail loudly）                                                                                         | 公式の主要機能が使えない                                                          |
| vision / audio  | 対応                                                                 | 未対応。checkpoint 2,780 テンソル中 1,179 本（vision_tower 546・audio_tower 631・embed_vision 1・embed_audio 1）が丸ごと未使用 | マルチモーダル不可                                                                |
| MTP drafter     | 対応（drafter は別 repo）                                            | 型と実行時で拒否                                                                                                               | 速度                                                                              |

出典: 公式 LiteRT-LM モデルカード、technical report（arXiv 2607.02770 Table 3 が int8 KV 前提で 32k を見積もる）、
`transformers/integrations/gemma_quant.py`、`transformers/quantizers/quantizer_gemma.py`、
`chat_template.jinja`、checkpoint の safetensors ヘッダ実測。
technical report の int8 KV 見積り（+0.05 GB）が karume の試算（99 MiB）の約半分になる理由は未追跡。

## 9. 構造門と恒等 SRQ

構造門 `admitGemma4Qat` に実グラフの改変プローブ 6 種（偽 projection の追加・片側 SRQ・head 格納の
差し替え・PLE 次元不一致など）を当て、全て拒否された。`scale = 0` の SRQ（実配布形に 2 本）は正しく通す。

公式 checkpoint で SRQ scale が 0（未較正 = 恒等）なのは **lm_head の入出力 2 本だけ**で、
IR にはそれが `[1,R,1536]` と `[1,R,262144]` の `static_quantize` として残る。
後者は decode 1 step ごとに語彙全体（約 1 MiB）を読んで同じ値を書き戻す恒等 pass になる
（262,144 要素の単体時間 0.0235 ms）。`i4-fast` の linear → SRQ 融合でも、
head は `i2` の n = 262,144 が並列 GEMV の許可表に無いため融合されず恒等 dispatch が残る。

## 10. 取り下げた主張（レビュー内で反証されたもの）

- **「窓 512 を跨ぐ QAT 実行が 1 件も無い」は事実に反する**。
  [WebML 比較の速度ベンチ](2026-09-12-webml-browser-speed.md)に 1,024 長の QAT 実行があり、実 GPU で完走している。
  残るのは「品質を見た実行が無い」で、検収成果物の最長入力は 65 token。
- **「capacity 8192 の実行実績が無い」も事実に反する**。
  [prefill バケット調査](2026-09-13-prefill-buckets.md)が容量 8192 の QAT を実走し、標準設定と token ID・停止理由が一致している。
- **「公式は bf16 なのに karume は f32」は棄却**（§5）。
- **「karume の RoPE 逆周波数が数学的に正しい」は訂正**（§3 — 正しいのはべき乗段だけ）。
- **「perf-ledger に固定 QAT の行が無い」は事実に反する**（K-27 が状態 ✅ で実在する）。
- **「SRQ のビット幅 8 を recipe が検査していない」は不足に当たらない**
  （公式側も唯一の呼び出し元が bits を渡さず、config も 8 に固定している）。

## 11. この記録で埋まっていないこと

- 実 GPU での挙動（WGSL の実行・並列 GEMV の実加算順・`i4-fast` の実走・512 超の文脈での出力品質・長文の速度）。
  全レッグが GPU 禁止で動いたため。
- 公式 checkpoint 全体を bf16 でロードしたときの生成列（公式推奨が f32 に解決するので優先度は低い）。
- SRQ ノード 65 本の共通部分式の畳み込みが converter のどこで起きるか（結果の等値は確認済み・実装箇所は未追跡）。
