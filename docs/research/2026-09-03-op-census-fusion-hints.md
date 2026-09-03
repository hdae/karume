# OP census と Fusion 候補列挙 — 8 家族の実資産に掛けた結果（2026-09-03）

> **性格**: 時点スナップショット（2026-09-03・GPU 不要の静的解析 — 配布ミラー 5 + 系列出力 5 の IR）。
> OP マイクロベンチ波（backlog ②）と Fusion 半自動発見（同 ③）の **1 段目**の成果で、道具は
> `tools/opbench`（census）と `tools/fusion-hints`（候補列挙）。生データは
> `outputs/bench/<資産>/2026-09-03_{op-census,fusion-hints}/`（git 追跡外）。数値は「実行 1 回・
> コンポーネント合算」で、anima は DiT を steps 倍・VAE を 9 タイル倍しないと予測 1 回にならない。
> GPU 時間の順位付けはまだ無い（2 段目 = 単体マイクロベンチ・e2e census との突合）。

## §1 census（tools/opbench）

「本数」= その op の IR ノード本数（≒ dispatch 数）、「出力要素」= そのノード群が書く出力テンソルの要素数合計（f32 なら ×4 バイト）。\
出どころ = `outputs/bench/<資産>/2026-09-03_op-census/summary.json`。

| 家族 / 資産                         | シナリオ（束縛）                 | 総ノード | 上位 op（本数）                                                   | 出力要素合計 |
| ----------------------------------- | -------------------------------- | -------- | ----------------------------------------------------------------- | ------------ |
| gemma4 / karume-gemma4-e2b          | decode（M=1, C=4096）            | 1,500    | linear 277 / rms_norm 242 / mul 209 / add 156 / reshape 142       | 4.3M         |
| gemma4 / karume-gemma4-e2b          | prefill（M=768, C=4096）         | 1,500    | 同上（M 非依存）                                                  | 2,489.7M     |
| anima / karume-anima                | 1024px（S=4096, T=64, Ttgt=512） | 5,311    | reshape 1,041 / linear 711 / mul 698 / add 592 / slice 530        | 17,046.9M    |
| irodori / karume-irodori-v4-small   | representative（T=256, S=750）   | 3,421    | mul 608 / reshape 532 / linear 505 / add 414 / slice 374          | 35,463.9M    |
| irodori / karume-irodori-v4.1-small | 同上                             | 3,421    | 同上（v4 と完全一致）                                             | 35,463.9M    |
| sbv2 / karume-sbv2-jvnv             | p203（P=203, T=203/35）          | 3,815    | reshape 864 / permute 526 / add 333 / **conv1d 321** / expand 277 | 589.7M       |
| siglip2 / so400m-patch14-384        | native（記号なし）               | 984      | reshape 282 / linear 168 / permute 140 / expand 113 / bmm 56      | 1,965.8M     |
| siglip2 / base-patch16-224          | native                           | 459      | reshape 132 / linear 78 / permute 65 / expand 53 / bmm 26         | 95.1M        |
| birefnet / birefnet-hr-1024         | native                           | 3,249    | reshape 991 / slice 468 / permute 331 / add 214 / cat 213         | 13,993.7M    |
| depth-anything / v2-small-hf        | native                           | 541      | reshape 132 / linear 72 / permute 69 / expand 49 / conv2d 33      | 1,004.0M     |
| vowel-detector / crnn-epoch3        | voiced（T=100）                  | 18       | linear 5 / permute 3 / conv1d 2 / gru_scan 2                      | 0.4M         |

NOTE: anima は 1 コンポーネント 1 run ぶん。1024px の予測 1 回は DiT を steps 倍・VAE decoder を 9 タイル倍する。

### §1.2 P-5（strided 実体化の consumer 畳み込み）の起票根拠

P-5 の kill 基準は「kind 別最大の 1 kind が 100ms（全 GPU 1.3%）未満なら記録のみ」。census が与えるのはその**上流の実数**で、「本当にバイトを書く strided ノード」だけを数えた（`reshape` は常に別名化 = 0 dispatch、恒等 `expand` も別名化なので除外し、融合に飲まれたものも除外）。既存の起票文の「graph 由来 6,520 dispatch ≈303ms」は外挿だったが、以下は資産から読んだ実数。

| 家族 / シナリオ                   | 実体化ノード | 出力要素 | f32 換算の書き出し | kind 別の最大                           |
| --------------------------------- | ------------ | -------- | ------------------ | --------------------------------------- |
| birefnet / native                 | 1,012        | 3,097.3M | **11,815 MiB**     | cat 213 本 / 1,075.3M el                |
| anima / 1024px（DiT 1 step ぶん） | 847          | 1,550.9M | 5,916 MiB          | **permute 454 本 / 1,518.6M el（98%）** |
| irodori / representative          | 589          | 400.7M   | 1,529 MiB          | cat 112 本 / 160.1M el                  |
| gemma4 / prefill（M=768）         | 285          | 285.3M   | 1,088 MiB          | permute 100 本 / 139.2M el              |
| siglip2 so400m / native           | 140          | 115.9M   | 442 MiB            | permute 140 本 / 115.9M el（100%）      |
| sbv2 / p203                       | 808          | 65.5M    | 250 MiB            | permute 526 本 / 61.0M el               |
| depth-anything / native           | 74           | 40.0M    | 152 MiB            | permute 69 本 / 37.3M el                |
| **gemma4 / decode（M=1）**        | 240          | **0.4M** | **1 MiB**          | permute 100 本 / 0.2M el                |

読み: strided 実体化は**画像系 1 本立ち**で、しかも kind が家族ごとに違う（anima と siglip2 は permute ほぼ 100%、birefnet は cat / slice / permute が三つ巴、irodori は cat 先頭）。LLM decode（gemma4 M=1）は 1 MiB で、P-5 の改造対象として意味を持たない。P-5 を「permute の consumer 畳み込み」に絞れば anima / siglip2 の実体化の 98〜100% を 1 種類の改造で取れる、というのが census から出る新しい事実。

### §1.3 K-1b（DiT attention の次の道具）の起票根拠 — anima transformer

融合で消えず素の attention op として出る本数と、その厳密な形（`fused_by: null` を確認済み）:

| コンポーネント   | op                      | 入力 shape                             | 本数（run 1 回）  |
| ---------------- | ----------------------- | -------------------------------------- | ----------------- |
| transformer      | attention（self）       | Q/K/V すべて [1,16,4096,128]           | 28                |
| transformer      | attention（cross）      | Q [1,16,4096,128] / K,V [1,16,512,128] | 28                |
| vae_decoder      | attention               | [1,1,4096,384] ×3                      | 1                 |
| text_encoder     | bmm + softmax（分解形） | [16,64,64]×[16,64,128] ほか            | 28+28、softmax 28 |
| text_conditioner | bmm + softmax（分解形） | [16,512,512] を経由                    | 24、softmax 12    |

読み: K-1b の対象は **DiT の 56 本（self 28 + cross 28）で頭幅 128 の 1 形だけ**。8 step の予測 1 回では 448 dispatch。self と cross で K/V 長が 4096 対 512 と 8 倍違うので、単体スパイクは**2 形を別々に測る**必要がある（1 形だけ測って倍率を一般化すると、もう片方に効かない道具を採用しうる）。

### §1.4 K-4b（SBV2/dacvae conv1d の i8a8 化）の起票根拠 — sbv2 F1 / i8+bert4

| コンポーネント | conv1d 本数 | 出力要素   | 格納            |
| -------------- | ----------- | ---------- | --------------- |
| voice          | 245         | 136.1M     | 全て f32+i8     |
| front          | 75          | 3.5M       | 全て f32+i8     |
| text_encoder   | 1           | 0.0M       | f32+i8          |
| **計**         | **321**     | **139.6M** | **f32+i8 100%** |

上位の形（本数順）: `[1,192,203]×[192,192,1]` 96 本（voice の 1×1）/ `[1,192,203]×[192,192,1]` 38 本（front）/ `[1,192,203]×[768,192,5]` 24 本 / `[1,768,203]×[192,768,5]` 24 本 / `[1,128,12992]×[128,128,3]` 6 本（codec 側の長尺）。

読み: **重み側はすでに 100% i8 格納**なので、K-4b が残しているのは活性側だけ（f32 → a8）。本数の 3/4 は voice に集中し、そのうち 96 本が kernel 1 の 1×1 conv = 実質 GEMV。**1×1（kernel 1）と kernel 3/5 は別カーネル形なので、i8a8 化の単体スパイクは最低この 2 群に分けて測る**。長尺の `[1,128,12992]` 6 本だけが桁違いに大きく（1 本あたり 1.66M el）、他は 0.04M el 級 — 加重の重心が本数と一致しない唯一の家族。

### §1.5 格納 dtype の分布（w8/w4 の実配布での実効範囲）

| 家族                           | 格納シグネチャ別ノード数                                                           |
| ------------------------------ | ---------------------------------------------------------------------------------- |
| gemma4（i4）                   | none 940 / f32 282 / **f32+i4g32 276** / i8 1 / f32+i8 1                           |
| anima（f16+dit8-a8-attn8-s16） | none 3,975 / f32 586 / **f32+i8 454** / f16+f32 294 / f16 2                        |
| irodori（i8-a8）               | none 2,418 / **f32+i8 567** / f32 435 / i8 1                                       |
| sbv2（i8+bert4）               | none 2,929 / f32 367 / **f32+i8 332** / **f32+i4g32 176** / i32 7 / i8 3 / i4g32 1 |
| 未配布 4 家族                  | 量子化ゼロ（f32 / none のみ）— 系列出力は未量子化                                  |

読み: 圧縮格納が乗っているノードは各家族 8〜15% で、残りは elementwise（`none`）と f32 の正規化・bias。「重みの量子化で GPU 時間が下がる」の効く範囲がここに実数で出る。

### §1.6 融合の静的ヒット数（既設の門との突合 — 全一致）

| 家族 / シナリオ         | ヒット（融合ステップ本数）                                        | 別名化（0 dispatch） |
| ----------------------- | ----------------------------------------------------------------- | -------------------- |
| gemma4 decode（M=1）    | rope 15                                                           | 142                  |
| gemma4 prefill（M=768） | **なし**                                                          | 142                  |
| anima 1024px            | identityExpand 160 / rope 112 / adaln 85 / silu 59 / upsample2x 3 | 1,189                |
| irodori representative  | rope 50 / silu 32 / identityExpand 24 / rowBlockAttention 12      | 520                  |
| sbv2 p203               | identityExpand 210                                                | 1,074                |
| birefnet                | identityExpand 192                                                | 1,183                |
| siglip2 so400m / base   | identityExpand 113 / 53                                           | 395 / 185            |
| depth-anything          | identityExpand 49                                                 | 181                  |
| vowel-detector          | なし                                                              | 0                    |

anima・irodori・gemma4 は `packages/runtime/tests/assets_fusion_counts_test.ts` のコンポーネント別期待値の合計と完全一致（anima: rope 56+56、adaln 85、silu 2+28+29、identityExpand 112+48）。**未配布 4 家族と sbv2 は融合ルールを 1 つも掴んでいない**（identityExpand は融合ではなく別名化の観測）— sbv2 の conv1d 主体・画像系の conv2d 主体という構成上の当然の帰結だが、③ Fusion 半自動発見の探索先としては「まだ 1 本も掴めていない 5 資産」が最大の未開拓面である、という事実が census から直接読める。

## §2 融合候補（tools/fusion-hints）

> **採取時点**: §2 の数値は 06:10 の掃引（生データ `candidates.jsonl` の mtime）で、道具の堅牢化
> コミット `e23f8b2`（08:37 — 短い read の検査・数値引数の検査・ソートのタイブレーク）より前。
> 最終版の道具で採り直しておらず、差の有無は未確認。\
> **訂正（2026-09-03）**: §2.2 の偶奇 RoPE と §2.3 の分解 attention の数え方を生データから採り直して
> 訂正した。誤りの機序は 2 つ — ① 窓幅の既定 9（`tools/fusion-hints/main.ts`）が 9 ノードを超える鎖を
> 切り詰めるので、同じ構造が資産ごとに違う op 名列として現れ、別綴りの行を合算していた ② 完全一致形
> （`bmm,reshape,softmax,…`）とマスク加算 `add` を挟む形を区別していなかった。

### §2.1 合格線（融合を切った計画 → 既知のヒット数が候補として再現）

テスト `tools/fusion-hints/enumerate_test.ts` が実資産に対して固定（列挙器そのものの合成 IR による単体は `packages/runtime/tests/runtime_fusion_hints_test.ts` 側に残る）。左が融合を切った計画での候補本数、右が現行計画（融合あり）で残る本数。

| 資産 / グラフ            | 鎖の op 名列                                                     | 融合オフ | 現行計画 |          出典の既知ヒット数 |
| ------------------------ | ---------------------------------------------------------------- | -------: | -------: | --------------------------: |
| anima transformer (DiT)  | `slice,slice,neg,cat,mul,mul,add`                                |       56 |        0 |                     rope 56 |
| anima transformer        | `layer_norm,reshape,reshape,add,mul,add`                         |       85 |        0 |                    adaln 85 |
| anima transformer        | `sigmoid,mul`                                                    |        2 |        0 |                      silu 2 |
| anima text_encoder       | `mul,slice,slice,neg,cat,mul,add`                                |       56 |        0 |                     rope 56 |
| anima text_encoder       | `sigmoid,mul`                                                    |       28 |        0 |                     silu 28 |
| anima vae_decoder        | `sigmoid,mul`                                                    |       29 |        0 |                     silu 29 |
| Irodori DiT              | `bmm,reshape,add,safe_softmax,expand,reshape,expand,reshape,bmm` |       12 |        0 |        rowBlockAttention 12 |
| Irodori DiT              | `sigmoid,mul`                                                    |   **29** |   **12** | silu 17（差 12 = ゲート形） |
| Irodori backbone         | `mul,slice,slice,neg,cat,mul,add`                                |       50 |        0 |                     rope 50 |
| EmbeddingGemma-300m      | `mul,slice,slice,neg,cat,mul,add`                                |       48 |        0 |                     rope 48 |
| MiniCPM5 decode          | `mul,slice,slice,neg,cat,mul,add` / `sigmoid,mul`                |  48 / 24 |    0 / 0 |           rope 48 / silu 24 |
| Gemma 4 E2B decode (M=1) | `mul,slice,slice,neg,cat,mul,add`                                |       50 |   **35** |                     rope 15 |

- rope の 56（anima text_encoder）と 48（EmbeddingGemma）は**窓幅 8 の出現を含む** — cos / sin 表の `sym_prefix_slice` を窓内 passthrough として跨いだ形。鎖と passthrough を切り分けられないと 55 / 46 になるので、この 2 つが passthrough 処理の門になっている。
- Irodori DiT の silu だけ 29 ≠ 17。差の 12 は `mul(v, sigmoid(u))` のゲート（自分自身に掛からないので SILU_RULE の外）で、現行計画に掛けるとその 12 だけが残る。op 名列の n-gram はルールの受理集合より広い、の実例。

### §2.2 既知の穴が数字として出ているか

> 表の「例 node NN」は道具が出す `example` 欄で、**ステップ順の添字**である。IR のノード添字と
> 一致するのは `--no-fusion` のときだけ（融合ありの計画では融合ステップが 1 本に畳まれて添字が
> ずれる）。鎖の現物を引くときは添字ではなく `outputName` で探す。

| 既知の穴                                                                 | 出た数字（現行計画の候補）                                                                                                                                                                                                                                                         | 場所                                                                              |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| gemma4 decode の rope 50 鎖中 15 しか掴めない                            | `mul,slice,slice,neg,cat,mul,add,permute` = **35**（極大 35・窓幅 8）                                                                                                                                                                                                              | `outputs/bench/gemma4-e2b-decode/2026-09-03_fusion-hints/` 例 node 88 / `permute` |
| Irodori DiT の偶奇 RoPE（ROPE_RULE は半割り形だけ受理）                  | `neg,slice,cat,reshape,mul,mul,add` = **24**（12 ブロック × q/k・窓幅 7）。先頭に `slice` を足した窓幅 8 の `slice,neg,slice,cat,reshape,mul,mul,add` は 23（24 本のうち 1 本だけ直前の op が `slice` でない）で、極大は窓幅 9 の `slice,neg,slice,cat,reshape,mul,mul,add,cat` 23 | `karume-irodori-v4-small/…` v4-small/dit                                          |
| Irodori DiT の rms_norm ベース adaLN（ADALN_RULE の先頭は `layer_norm`） | `rms_norm,add,mul,add` = **24**、拡張形 `add,rms_norm,add,mul,add` = 23                                                                                                                                                                                                            | 同上                                                                              |
| （同族）Irodori speaker の偶奇 RoPE                                      | `slice,neg,slice,cat,reshape,mul,mul,add` = **15**（+ `neg,slice,cat,…` 形 8）                                                                                                                                                                                                     | v4-small/speaker                                                                  |

### §2.3 家族 × 上位候補（現行計画・極大順・別名化のみの鎖は除外）

| 家族 / グラフ           | ノード | 相異なる鎖 | 上位候補（極大本数）                                                                                                                                                                    |
| ----------------------- | -----: | ---------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| anima transformer       |   2603 |         69 | `permute,attention,permute,reshape,linear,mul,add` 56 / `linear,reshape,permute,rms_norm` 112 / `linear,mul,add` 84                                                                     |
| anima text_encoder      |   1741 |        122 | `add,softmax,expand,reshape,expand,reshape,bmm,reshape,permute` 28 / `linear,reshape,rms_norm,permute` 56 / `linear,add` 55                                                             |
| anima text_conditioner  |    589 |        140 | `mul,slice,slice,neg,cat,mul,add` **24**（rope 綴りなのに未掴） / `linear,reshape,rms_norm` 24 / 分解 attention（完全一致形）12                                                         |
| anima vae_decoder       |    378 |         86 | `mul,sum,sqrt,clamp_min,reshape,div,mul,mul` 29（weight-norm 相当）                                                                                                                     |
| gemma4 e2b decode       |   1570 |        126 | `linear,rms_norm,add` 70 / `mul,linear,rms_norm,add` 35 / `linear,gelu_tanh,linear,mul,linear,rms_norm,add` 35 / rope 残り 35                                                           |
| irodori dit             |   1465 |        119 | `linear,reshape,rms_norm,mul` 36 / 偶奇 rope 23（極大形 `slice,neg,slice,cat,reshape,mul,mul,add,cat`・出現本数は 24 — §2.2） / `rms_norm,add,mul,add` 24 / `linear,linear,add,tanh` 24 |
| irodori backbone        |    957 |         32 | `attention,permute,reshape,linear,add` 25 / `slice,slice,gelu,mul,linear,add` 24 / `layer_norm,linear` 49                                                                               |
| irodori codec enc/dec   |    248 |    37 / 40 | `mul,sin` 29（Snake 活性） / `mul,mul,add,reshape,conv1d,reshape` 13 / 12                                                                                                               |
| sbv2 F1/voice           |   1788 |        302 | `conv1d,leaky_relu,conv1d,add` 35 / `bmm,reshape` 族 72 / `add,permute,layer_norm,permute` 40                                                                                           |
| sbv2 F1/text_encoder    |   1128 |         74 | `softmax,reshape,bmm,reshape,permute,reshape,linear,add,layer_norm` 22 / `add,layer_norm` 44                                                                                            |
| sbv2 F1/front           |    899 |        427 | `conv1d,permute,layer_norm,permute,gelu,conv1d,permute,layer_norm,permute` 12                                                                                                           |
| siglip2 so400m          |    984 |        135 | `bmm,reshape,softmax,expand,reshape,expand,reshape,bmm,reshape` **28** / `linear,reshape,permute,mul` 56                                                                                |
| birefnet-hr-1024        |   3249 |        545 | 分解 attention のマスク加算入り綴り `add,softmax,expand,reshape,expand,reshape,bmm,reshape,permute` **48** / `layer_norm,linear,gelu,linear,add` 46 / `slice,reshape,permute` 96        |
| depth-anything v2 small |    541 |        265 | 分解 attention（siglip2 と同じ完全一致形）**12** / `linear,mul,add` 23                                                                                                                  |
| vowel-detector CRNN     |     18 |         31 | `linear,gru_scan,linear,gru_scan_reverse,cat` 1（グラフが 18 ノードなので候補は各 1 本）                                                                                                |

**横断で最も本数が多い未掴の形**: 分解 attention。ただし**同じ構造でも op 名列は資産ごとに割れる**ので、1 行の本数を足し合わせると数を取り違える（訂正前の本文はそれをやっていた）。完全一致の `bmm,reshape,softmax,expand,reshape,expand,reshape,bmm,reshape` を持つのは siglip2 so400m 28 / depth-anything 12 / anima text_conditioner 12 の計 52 本。birefnet と anima text_encoder はマスク加算 `add` を挟む別綴りで、`bmm,reshape,add,softmax,expand,reshape,expand,reshape,bmm` が birefnet 12 / anima text_encoder 27（窓幅 9 で切れるため、同じ構造が `add,softmax,…,permute` のような別の窓としても出る）。

綴りに依らないブロック本数は、どの資産でも共通の核 `softmax,expand,reshape,expand,reshape,bmm` の出現数で数えられる — **5 グラフ（4 家族）計 128 本**（birefnet 48 / siglip2 so400m 28 / anima text_encoder 28 / anima text_conditioner 12 / depth-anything 12）。ROW_BLOCK_ATTENTION_RULE はこの綴りを `safe_softmax` でだけ受理するので全て外れている（Irodori DiT の 12 本のみが `safe_softmax` 側）。

### §2.4 注意（表の読み方）

- 順位付け（GPU 時間）はしていない。上位には `reshape,permute` のような別名化ノード対（0 dispatch）も混ざるので、② の census と突合するまでは「本数」以上の意味を持たない。
- n-gram は長い鎖の接頭辞も同じ本数で出す。読むのは `maximal`（同じ先頭からより長い窓が受理されなかった出現数）が大きい行で、表はその順に並べてある。
- 窓幅 > 鎖の長さの行は窓内 passthrough を含む（例: `mul,slice,slice,neg,cat,mul,add` の窓幅 8 = 表の `sym_prefix_slice` を跨いだ形）。

## §3 含意（2 段目への入口）

- **P-5 は「permute の consumer 畳み込み」に絞る**: anima / siglip2 の strided 実体化は permute が
  98〜100%。LLM decode は 1 MiB で対象外。
- **K-1b は 2 形を別々に測る**（DiT self 4096 / cross 512・頭幅 128）。**K-4b は 1×1 と kernel 3/5 の
  2 群**（voice の 96 本が実質 GEMV）。
- **融合の未開拓面は 5 資産**（sbv2・未配布 4 家族は融合ルールを 1 本も掴んでいない）。横断で最多の
  未掴形は分解 attention で、**構造のブロック本数**（共通核 `softmax,expand,reshape,expand,reshape,bmm`
  の出現数）で数えると 5 グラフ計 128 本。op 名列は資産ごとに割れるので、この 128 はブロック本数であって
  1 つの綴りの本数ではない（内訳は §2.3 末尾）。ROW_BLOCK_ATTENTION_RULE の受理を `safe_softmax` から
  `softmax` へ広げるかは裁定事項（ADR 0040 決定 2「式が似ていても広げない」との衝突）。
- **既知の穴が数字になった**: gemma4 decode の rope 50 鎖中 35 本未掴（`…,add,permute` の窓幅 8）・
  Irodori DiT の偶奇 RoPE 24 / rms_norm adaLN 24（K-7 の再評価材料）・anima text_conditioner の rope
  24 本（並ぶのに 0 本 — 機序未特定）。
- **起票（リリース後・道具側）**: ① `tools/fusion-hints` を `--max-window 10`〜`12` で採り直す — 既定 9
  では 9 ノードを超える鎖が切り詰められ、同じ構造が資産ごとに違う op 名列になって横断突合が効かない
  （§2.3 の分解 attention が実例）。② fusion-hints は siglip2-base-patch16-224 と
  karume-irodori-v4.1-small に掛かっていない（census は §1 のとおり掛かっている）— 家族内の大小 2 形の
  突合ができないままなので、掃引対象に足す。
- 2 段目（GPU）: `tools/opbench single`（計測規約を実装として内蔵・timing / wall 2 モード・K-11 と P-1
  の再現が合格線）→ `graph` → PyTorch 対照（CUDA venv・列 B「torch が実際に速い形」を基準・常駐
  バイト併記 — 2026-09-03 裁定）→ ブラウザ面 → CPU/TS 配置評価。Inductor は候補の裏付け役（2 段目）。
