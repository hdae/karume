# gemma4 E2B — chunkLength 掃引と K-12（③PV の KV 並列縮約）の実測（2026-09-03）

> **性格**: 時点スナップショット（2026-09-03・RTX 3080 Ti 12GB / Linux / Vulkan (wgpu) / Deno 2.9.6）。
> [P 掃引](2026-09-03-gemma4-context-length-sweep.md) の続編で、可変 capacity 波の設計材料 2 点 —
> ①chunkLength（prefill の刻み幅）を再 export なしで上げたときの prefill 時間、②decode の full 層 attention
> ③PV 段を KV 長方向に並列化した変種（perf-ledger K-12・`stateAttentionReduce: "parallel"`）の効き。
> 各点 1 回。台本と生データは `outputs/bench/karume-gemma4-e2b/2026-09-03_chunklength-sweep/`（git 追跡外）。

## §1 方法

- 資産は P 掃引と同じ計測専用配布形 `pos16k`（RoPE 表 16,384 行・他は出荷バイトと同一）。chunkLength は
  `karume.json` の `pipelineConfig.chunkLength` だけを書き換えた変種ディレクトリ（shard は symlink）で与える —
  出荷 IR に M の上下限は焼かれていない（記号は名前のみ・768 由来の定数ゼロ）ので**再 export は要らない**。
- 台本は P 掃引の `sweep.ts` に `--path session|pipeline` と `--reduce sequential|parallel` を足したもの。
  chunkLength 掃引は公開面（`Gemma4Pipeline.fromPretrained` + `denoDirectory`・timing off）、K-12 は低レベル面
  （Session を自分で組む — `SessionOptions.stateAttentionReduce` は公開面に席が無い・timing off）。
  op 別内訳は timing on（壁時計は読めない — 同 §1 の注意）。合成プロンプト・greedy・N = 32 / 48。

## §2 chunkLength 掃引（prefill 壁時計・decode は不変）

|      P | cl=32（[P 掃引](2026-09-03-gemma4-context-length-sweep.md)） | cl=256 | cl=512 | cl=768 |
| -----: | -----------------------------------------------------------: | -----: | -----: | -----: |
|    992 |                                                       19.2 s |  3.6 s |  2.4 s |  2.2 s |
|  4,096 |                                                         75 s | 13.3 s | 10.6 s |  8.3 s |
| 16,000 |                                                        326 s |   77 s |   71 s |   60 s |

- greedy の token 列（24 token）は cl=32 / 256 / 512 / 768 で一致（P=4,096・session 経路で cl=32 と 512 を突合）。
- 最終 chunk の GPU 内訳（P=4,096・timing on）: cl=32 は 146 ms（linear 91 / attention 46）、cl=512 は 1,225 ms
  （linear `reg64x32` 488 / attention 713 = full qk 268 + full pv 246 + sliding 196）。大きい chunk では prefill 時間が
  GPU 時間そのものになり、内訳は **attention（K/V を query 行ごとに読み直す traffic 律速）と M=512 GEMM（実効
  ≈5.5 TFLOPS）が半々**。
- 所見: 見積り（2〜3 倍）より効きが大きい（5〜9 倍）。理由は cl=32 の chunk あたり固定費（≈600 ms）が M に対して
  ほぼ不変だったこと。**製品の既定は 768**（export の Dim 上限 = 検証済み範囲の上端）。VRAM は S 一時
  `8 × M × C × 4 B` が主で、capacity 4,096 なら 96 MiB。

## §3 K-12 — ③PV の KV 並列縮約（cl=768・session 経路・壁時計）

|      P | prefill（sequential / parallel） | decode 中央値 ms/token（sequential → parallel） |  倍率 |
| -----: | -------------------------------: | ----------------------------------------------: | ----: |
|    992 |                    3.87 / 3.84 s |                                     39.7 → 32.6 | ×1.22 |
|  4,096 |                    12.4 / 12.6 s |                                     47.0 → 37.1 | ×1.27 |
| 16,000 |                    67.0 / 65.5 s |                                     81.3 → 41.0 | ×1.98 |

- token 列は 3 点とも sequential と一致（24 token）。prefill は不変（誤差内）。
- decode のキー別 GPU 時間（P=16,000・timing on・GPU 総和 68.0 → 38.8 ms）:

| キー                                  | sequential | parallel |
| ------------------------------------- | ---------: | -------: |
| `attention_state_pv` full（7 本）     |    35.0 ms |   3.6 ms |
| `attention_state_pv` sliding（28 本） |     6.2 ms |   0.8 ms |
| `attention_state_qk` full（7 本）     |     6.1 ms |   6.6 ms |
| `linear_gemv`（276 本）               |     9.4 ms |  12.9 ms |

- 所見: 機序は P 掃引 §3 の見立てどおり（③ は有効 invocation が `D × B·H` = 4,096 で固定・KV 長は 1 スレッドの逐次長）。
  16 レーン化で full PV ×9.7・sliding PV ×7.8。sliding は P に依らないので**短い P でも −7 ms/token** 効く。
  次の attention レバーは ①QK（D=512 の逐次内積・6.6 ms）。linear_gemv の揺れは timing on の測定代。
- 数値: ③' は縮約順が違うので ③ とビット同一ではない。A/B 帯門（`tests/gpu_state_attention_parallel_test.ts`）の
  実測最悪は vs ③ 2.38e-7 / vs f64 参照 3.99e-7（帯 5e-6）。決定性・容量非依存・行ブロック非依存・pad 行厳密 0 は
  ビット門で保持。opt-in 席 = `stateAttentionReduce`（既定 `"sequential"` — ADR 0058）。

## §4 新配布形（RoPE ホスト供給・chunkLength 768 / capacity 4,096 既定）での再測

再 export 後のミラー `models/karume-gemma4-e2b`（ADR 0091）で同じ台本を回した（session 経路・N = 48・
capacity は P に合わせて上書き）。

|      P | capacity | prefill（seq / par） | decode ms/token（seq → par） |
| -----: | -------: | -------------------: | ---------------------------: |
|    992 |    4,096 |        3.99 / 3.91 s |                  40.0 → 33.0 |
|  4,096 |    4,160 |        12.2 / 12.6 s |                  47.1 → 34.7 |
| 16,000 |   16,384 |        68.2 / 74.0 s |                  82.2 → 41.1 |

token 列は 3 点とも旧計測用配布形（pos16k）と一致 — 出荷バイト（GPTQ 再走）と RoPE 表（f64 → f32）が
両方変わっても greedy の列は動かなかった。

`Gemma4Pipeline.estimateSessionMemory` の実値（同ミラー・RTX 3080 Ti の granted limit）:

| capacity |   weights | state（KV） | prefill workspace | peakAccounted |
| -------: | --------: | ----------: | ----------------: | ------------: |
|    4,096 | 1,506 MiB |      60 MiB |           431 MiB |     2,028 MiB |
|   16,384 | 1,506 MiB |     204 MiB |           719 MiB |     2,460 MiB |
|   65,536 | 1,506 MiB |     780 MiB |         1,871 MiB |     4,188 MiB |
|  131,072 | 1,506 MiB |   1,548 MiB |         1,871 MiB |     4,956 MiB |

workspace が 65,536 で頭打ちなのは states 形 attention の一時 S が行ブロック（binding 上限で等分）に
収まるため。128K の会話は 12 GB 級の device なら見積り上は載る（時間の壁は §2 / §3 のとおり）。

### §4.1 32K / 128K（参考値 — 別プロセスの負荷下）

- P=32,000（capacity 32,768・cl 768・K-12 on・停止 token 無効）: prefill 228 s・decode 56.6 ms/token（22 サンプル）。
  実行中に Opus レッグ 3 本（CPU）と別の対話プロセス（VRAM 2.2 GiB）が同居していたので壁時計は上振れの参考値。
  同条件の P=992 が 62.9 ms/token（無負荷時 33.0）だったことから、無負荷なら 45 ms/token 前後と見込む
  （①QK の線形項 — perf-ledger K-14）。合成 prompt（同文の繰り返し）は停止 token を有効にすると
  prefill 直後に EOS を出す（decode を測るには `--no-stop` が要る）。
- P=131,000（capacity 131,072・cl 768）は `GpuOutOfMemoryError`（run のエンコードと readback）で落ちた。
  ただし同時刻に VRAM 12 GB のうち 7.5 GiB を別プロセス（前回計測の残骸 5.3 GiB + 対話プロセス 2.2 GiB）が
  握っていたので、環境要因の疑いが強い。無負荷での再現（cl 768 / 192）は未実施 — 結果は追記する。
  見積り上は peak 4,956 MiB（§4）で 12 GB 機に載る。
