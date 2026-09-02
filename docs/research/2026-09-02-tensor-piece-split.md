# テンソル分割（piece）の成立性スパイク — GPU オフセット書き・変換の行分離・詰めの実資産適用

> **性格**: 時点スナップショット（2026-09-02・RTX 3080 Ti 12GB / Linux / Vulkan (wgpu) /
> Deno 2.9.6）。ADR [0090](../decisions/0090-shard-spec-v3-tensor-pieces.md) の設計を実装前に
> 3 点で確かめたもの。数値は本機依存で、Mac（Metal）/ Chrome（Dawn）は未測。

## 前提 — 実資産で 256MiB を超えるテンソル

5 ミラー（anima / gemma4 / irodori v4・v4.1 / sbv2-jvnv）の全 shard ヘッダを読んだ結果、上限
256MiB を超えるテンソルは 3 本で、いずれも「語彙 × 隠れ次元」の埋め込み表。3 本とも生バイトのまま
GPU 常駐する席（raw / f16 / i8）で、CPU 展開席には該当が無い。

| コンポーネント       | テンソル                                 | 格納       | shape          |   サイズ | 席  | 消費 op           |
| -------------------- | ---------------------------------------- | ---------- | -------------- | -------: | --- | ----------------- |
| gemma4 model.i4      | `model.lm_head.weight`                   | i8 + scale | [262144, 1536] | 384.0MiB | i8  | embedding, linear |
| anima text_encoder   | `model.embed_tokens.weight`              | f16        | [151936, 1024] | 296.8MiB | f16 | embedding         |
| irodori backbone f32 | `model.embeddings.tok_embeddings.weight` | f32        | [102400, 768]  | 300.0MiB | raw | embedding         |

## 1. GPU — `queue.writeBuffer` のオフセット書き（384MiB・各 3 回）

1 本の 384MiB バッファへ「1 回書き」「2 piece（255MiB + 129MiB）」「8 piece（48MiB 刻み）」
「末尾 4 バイトだけの piece」で書き、submit → `onSubmittedWorkDone` まで計測。全変種で validation /
out-of-memory の errorScope は無音、読み戻し（`mapAsync`）は元データと u32 単位で全一致。

| 書き方   | issue（writeBuffer 発行） | fence（完了待ち） |     合計 |
| -------- | ------------------------: | ----------------: | -------: |
| 1 回書き |              44.7〜47.4ms |      19.6〜22.6ms | 64〜70ms |
| 2 piece  |              44.0〜45.0ms |            20.7ms | 65〜66ms |
| 8 piece  |              40.5〜46.8ms |      20.7〜22.4ms | 61〜69ms |
| 末尾 4B  |              44.1〜45.4ms |      21.6〜22.4ms | 66〜68ms |

所見: piece の本数は転送時間に乗らない（総バイト数で決まる）。台本 =
`scratchpad/gpu_piece_write.ts`（使い捨て・リポ外）。

## 2. 格納変換の行分離（ビット比較）

`_convert_for_storage`（f16 / i8 / i4）について「先頭次元で切ってから変換」と「変換してから同じ
行範囲を切る」のバイト列を比較。scale は `scale.shape[0] == tensor.shape[0]` のときだけ同じ行範囲で
切る規則。

| 形                                                | 一致 |
| ------------------------------------------------- | ---- |
| f16                                               | ✓    |
| i8・行 scale `[rows, 1]`                          | ✓    |
| i8・列 scale `[1, cols]`（scale は無切断）        | ✓    |
| i4・group scale `[rows, cols/16]`（rank 2）       | ✓    |
| i4・rank 3 `[O, Cin, K]`（scale `[O, Cin·K/16]`） | ✓    |

所見: 3 変換とも要素ごと（scale は行 or 列で broadcast）なので、行スライスと可換。書き手は piece を
`tensor[begin:end]` から変換して書けばよく、全体を変換してから切る必要が無い（同時に生きる変換済み
テンソルは piece 1 本ぶんに減る）。

## 3. 詰め — 既存 packer に 1MiB 行ブロックを流す（5 ミラー・95 コンポーネント）

上限超えの単位（weight + scale）の重みだけを 1MiB の行ブロック（4 バイト整列になる行数へ切り上げ）
に砕き、`pack_shards`（最小本数 → 均し）へ通常の単位として流した結果。容量 = 256MiB − 1MiB。

| コンポーネント        | 今（本数 / 最大 data） | 後（本数 / 最大 data） | 最大ファイル長（ヘッダ推定込み） | piece                 |
| --------------------- | ---------------------: | ---------------------: | -------------------------------: | --------------------- |
| anima text_encoder    |           6 / 296.8MiB |           6 / 232.8MiB |                        232.75MiB | embed_tokens 2 分割   |
| gemma4 model.i4       |           6 / 385.0MiB |           7 / 254.1MiB |                        254.07MiB | lm_head 2 分割        |
| irodori v4.1 backbone |           7 / 300.0MiB |           6 / 252.0MiB |                        252.05MiB | tok_embeddings 2 分割 |
| irodori v4 backbone   |           7 / 300.0MiB |           6 / 252.0MiB |                        252.05MiB | tok_embeddings 2 分割 |
| 他 91 コンポーネント  |               ≤ 255MiB |                   不変 |                                — | なし                  |

発見（設計へ反映済み）: 対の引き寄せを現行規則（片方に到達した時点で相方を 1 単位へ）のまま
ブロックに適用すると、F32 の scale が並びの先頭群に居るため**ブロック 1 だけが前へ引き寄せられ**、
残りが末尾群に取り残されて piece が shard [1, 5, 6] に散った。分割テンソルでは
「scale + 全ブロック」を鎖として引き寄せる規則が要る（適用後は全て連続 2 shard に収まった）。

もう 1 点（設計へ反映済み）: `requiredLimits` の導出（`karume.limits.max_tensor_payload`）は shard
ヘッダの最大テンソル長を採るので、piece 化すると親の全体長より小さい値が焼かれる。piece を親名で
合算する必要がある。

台本 = `scratchpad/pack_spike.py` / `convert_spike.py`（使い捨て・リポ外）。

## 4. 実装後のホスト RAM ピーク（Linux パイプライン面・各 3 回・`tools/ram-peak/measure.ts`）

再生成したミラー（piece 入り・全 shard ≤ 256MiB）で、Phase B の最終形（結果 7 — 目標 256MiB・実効目標で
持ち上がった shard あり）と同じ台本を再測定。

| 構成                                  | 最大 shard（前 → 後）                        | 前（結果 7） |           後 |   差 |
| ------------------------------------- | -------------------------------------------- | -----------: | -----------: | ---: |
| gemma4 i4（e2b・chat 最小生成）       | 385 → 254MiB                                 |      745 MiB | 611〜615 MiB | −18% |
| anima f16（turbo v1.1・512²・2 step） | 297 → 252MiB（text_encoder は 297 → 233MiB） |      766 MiB | 710〜712 MiB |  −7% |

所見: 「定数 + 最大 shard 1 本」のとおり、最大 shard の減少分（gemma4 −131MiB / anima −45〜64MiB）が
そのままピークから消えた。ロード・生成時間は不変（anima 生成 5.8〜6.0 s・gemma4 ロード 1.8〜2.2 s）。

Mac（Apple M2 / Metal / Deno 2.9.4・ユーザー実測）: 同じ台本で gemma4 の最大 RSS は 841 MiB（計測器）/
915 MiB（`/usr/bin/time -l`）。Mac の gemma4 は初計測で比較対象は無い。piece 入りミラー（`lm_head` 2 piece・
7 shard）は `e2e_gemma4_pretrained_test` / `e2e_gemma4_directory_test` の chat golden（温度 0 で fromAssets 経路と
同じ token 列）が緑 — Metal でもビット同一に読めている。
