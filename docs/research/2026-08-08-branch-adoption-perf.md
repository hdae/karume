# 参照実装ブランチからの再実装（C 波）— 採否と E2E 実測

> NOTE: 時点スナップショット。数値は 2026-08-08 の実機実測（RTX 3080 Ti 12,288MiB /
> Deno 2.9.4 / wgpu 29.0.1 / x86_64-linux）に基づく。参照実装ブランチ側の数値は
> [2026-08-06-kernel-triage/](2026-08-06-kernel-triage/) の記録（Apple M2 / Metal と
> RTX 3080 Ti / Vulkan）からの引用で、本リポでの再実測ではない。

参照実装ブランチ `codex/kernel-quick-fixes` の triage（KQF-* / QUANT-* / OP-*）を、本リポの
main へ**設計から書き直して**取り込んだ波（C1〜C3）の記録。採ったもの・採らなかったもの・
その実測根拠を残す。ブランチのコードをそのまま merge していないので、triage の主張は
「候補と信号の出どころ」であって本リポの実測ではない。

## 1. 計測プロトコル（全 A/B 共通）

出典ログ: `perf-pre-c-ff78553.log` / `perf-c2-head.log` / `perf-c2-f16.log` /
`perf-c3-tree.log` / `perf-final-head.log`（スクラッチパッド・生値）。台本は
`perf-measure2.zsh`。

- 計測対象は `packages/models/tests/e2e_anima_test.ts` の E2E 生成（`[e2e]` 行の壁時計）で、
  PNG の sha256 も同時に出る。preset / 解像度ごとに **3 run**。
- **クールダウン規約**: 開始前に 60 秒待って**アイドル温度**を確定し、各 run の前に
  「温度 ≤ アイドル + 5°C」になるまで待つ（上限 180 秒）。run ごとに pre / post の
  温度と SM クロックをログへ残す。
- **全 A/B を同日に実施**した。アイドル温度は 42〜45°C、各 run の pre は 42〜50°C /
  210MHz、post は 57〜82°C で、水準は全ログで揃っている。これで「日をまたいだ冷却状況の
  改善が速度差に化ける」交絡を排除している（下の A/B 表の差は温度水準の差では説明できない）。

## 2. 採用したもの

| #  | 内容                                                                                                         | commit    | 参照 triage              |
| -- | ------------------------------------------------------------------------------------------------------------ | --------- | ------------------------ |
| C1 | **i8a8 linear の accumulator 静的展開** — `acc[i]` の動的添字を `acc0`〜`acc3` の名前付き変数へ codegen 展開 | `4b15ec2` | KQF-005                  |
| C1 | **PipelineCache が未決着の生成を同一キーで共有** — 生成中の `Promise` 自体をキャッシュし、失敗時だけ削除     | `c528224` | KQF-001                  |
| C1 | **隣接 permute の合成と恒等 permute の除去**（exporter `normalize.py`）                                      | `5035f0e` | KQF-003                  |
| C2 | **融合パス新設** — RoPE / SiLU / upsample2x を 1 dispatch へ + 恒等 expand の別名化                          | `39d5d62` | OP-007 / 017 / 018 / 019 |
| C3 | **f32/f16 linear の accumulator 静的展開**（C1 と同じ機序の f32/f16 版）                                     | `df148dd` | KQF-004                  |

いずれも**設計の再実装**であって移植ではない。融合パスの構造（純関数パス・exact 一致・
常設カウンタ・exporter と runtime の役割分担）は ADR [0040](../decisions/0040-fusion-pass.md)
で確定させた。

数値不変の根拠:

- i8a8 の静的展開は縮約が i32 の厳密加算のままなので**返る整数が 1 ビットも変わらない**。
  K タイル 16・`kk` 昇順・1 出力あたりの加算順序・fma 単一丸め・`xs·wscale` の畳み順は不変。
- f32/f16 版も K タイル 16・`kk` 昇順・加算順序・bias の足し順が不変。共有骨格を使う他 5 op
  （matmul / bmm / attention ×2 / conv2d）の生成バイト列は 1 バイトも動いていない。
- **PNG sha256 門 4 本が全て参照値と一致**（既定 preset の w8a8-s16 は i8a8 経路、f16 preset は
  f32/f16 経路をそれぞれ通る）。

## 3. 確定 A/B（同日・3 run の生値）

| 条件                | before               | after                | 倍率      |
| ------------------- | -------------------- | -------------------- | --------- |
| **w8a8-s16 / 1024** | 16.1 / 16.1 / 16.1 s | 13.9 / 13.9 / 13.9 s | **1.16×** |
| **w8a8-s16 / 512**  | 4.7 / 4.7 / 4.7 s    | 4.1 / 4.2 / 4.2 s    | 約 1.13×  |
| **f16 / 1024**      | 38.5 / 38.8 / 38.8 s | 30.1 / 30.3 / 30.3 s | **1.28×** |

- w8a8 の before は C 波直前の main（`ff78553`）、after は最終 HEAD（`df148dd`）。
  中間の C2 時点（`perf-c2-head.log`）で既に 13.8 / 13.9 / 13.9 s・4.1 / 4.2 / 4.2 s に
  到達しており、C3 は w8a8 経路に影響しない（i8a8 linear を通るため）。
- f16 の before は C2 時点（`perf-c2-f16.log`）で、after は C3 の作業ツリー
  （`perf-c3-tree.log` = 30.2 / 30.2 / 30.2 s）と最終 HEAD（`perf-final-head.log`）。
  **f16 の 1.28 倍は C3 単独の効果**である。
- PNG sha256 は全ログの全 run で同一（w8a8-1024 `aa013054…`、w8a8-512 `dd4506de…`、
  f16-1024 `6943b541…`）。速度が変わってもビットは動いていない。

## 4. 融合ヒット数の実測

`Diagnostics.lastRunFusions`（ADR 0040 §3）の実測値。

| グラフ                          |                                         rope |    silu | upsample2x | identityExpand |
| ------------------------------- | -------------------------------------------: | ------: | ---------: | -------------: |
| anima 1024 / 8step / guidance=1 | **503**（transformer 448 + text encoder 55） | **305** |     **27** |        **160** |
| SBV2                            |                                            0 |   **0** |          0 |        **210** |

- anima 側は参照実装ブランチの静的集計（OP-007 の 448 + 55、OP-018 の 305 鎖、
  OP-019 の 160 本）と一致する。
- **SBV2 は sigmoid を 1 本も持たないため SiLU 融合の対象がゼロ**で、効くのは恒等 expand の
  別名化 210 本だけ。融合 on / off で WAV と dump が sha256 完全一致することを確認した
  （融合が「値を変えない」ことの、PNG 門とは独立した 2 本目の証拠）。

## 5. 採らなかったもの（実測根拠つき）

### 5.1 contiguous elementwise（KQF-002）— 壁時計に出ない

全入力が出力と同 shape のとき、broadcast のための「線形 index → rank 座標 → 入力 stride →
線形 index」の往復を省き、`inK[i]` で直読みする変種。参照ブランチの静的集計では候補が
elementwise 全体の **61.9%**（996 / 1,610）と大きく、有望に見えた。

本リポでも C3 と同波で実装し、**含む作業ツリー（`perf-c3-tree.log`）と除いた最終 HEAD
（`perf-final-head.log`）を同日・同プロトコルで比較した**:

| 条件            | contiguous あり      | contiguous なし               |
| --------------- | -------------------- | ----------------------------- |
| w8a8-s16 / 1024 | 13.8 / 13.9 / 13.9 s | 13.9 / 13.9 / 13.9 s          |
| w8a8-s16 / 512  | 4.1 / 4.2 / 4.1 s    | （C2 時点 4.1 / 4.2 / 4.2 s） |

**3 run とも完全同値**で、壁時計の効果が観測できなかったため**不採用**とした。
機序: elementwise は帯域律速であり、添字算術（u32 の `/` と `%`）の削減は DRAM 待ちの陰に
隠れて表に出ない。参照ブランチ側も「Metal 実機での壁時計改善率は未計測」と断っており、
静的な候補数は壁時計の予測子にならなかった、というのがこの項目の教訓である。

### 5.2 QUANT-010（i8a8 linear の行量子化共有）— 参照実測でノイズ内

同じ activation を読む i8a8 linear の fan-out について、`quantize_rows` の出力（packed i8 +
per-row scale）を 1 run 内で共有する案。参照実装ブランチの実測:

- 実配布 transformer で linear の行量子化 **454 → 259 本 = 195 dispatch/predict 削減**。
  RTX 実測でも総 dispatch が 3,301 → 3,106 本と予測どおり 195 本減った。
- しかし**壁時計は動かなかった**: RTX の単発 timestamp で総 GPU 時間 1,104.7 → 1,108.4ms
  （ノイズ内・`quantize_rows` 合計は 56.43 → 51.67ms）。Apple M2 の Anima 1024 既定 run も
  625.2s → 626.9s で、差は約 +0.3% の誤差圏。
- transient peak は 712.55 → 713.05MiB（+0.50MiB）で悪化はしないが、得もない。

**dispatch 数の削減が壁時計に出ない典型例**として不採用。run 越えの cache は staged execution
（参照 large-designs の D / E）の ownership 設計とセットでなければ意味が無く、それは別件。

### 5.3 i8a8 GEMM のタイル幾何一般化（KQF-006〜009）— 参照実測で全案退行

「Apple GPU では 256 thread / M64×N64 が occupancy を制限しているのでは」という仮説に対し、
参照ブランチが 4 案を直接 A/B したが、**採用に値する候補が 1 本も残らなかった**
（全案・全形状で数値は bit 一致。速度だけの話）。

| 候補                         | RTX 3080 Ti（加重） | Apple M2（加重）                | 判定                                                                                                                     |
| ---------------------------- | ------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| KQF-006 M32×N64 / 128 thread | 0.846×              | 1.036×                          | 不採用（M2 の名目 1.036 は attention 1.123 / modulation 1.591 の改善で ffn-up 0.858 / cross 0.845 の退行を相殺しただけ） |
| KQF-007 M64×N32 / 128 thread | 0.949×              | 0.896×                          | 不採用（主要形状に採用根拠なし）                                                                                         |
| KQF-008 M32×N32 / 64 thread  | paired 0.807×       | paired 0.846×                   | 不採用（主要 3 形状すべて明確に退行）                                                                                    |
| KQF-009 K32（K タイル拡大）  | paired 0.974×       | paired 1.044× / 実時間比 0.919× | 不採用（M2 の paired は p10/p90 が 0.823〜1.106 と 1 を両側に跨ぐ）                                                      |

**1 を超えた 2 つの数字（M2 の 1.036 / 1.044）はどちらも分散が大きく、主要形状の退行を別形状で
相殺した結果**である。backend selector や autotune は「選ぶ価値のある候補が 1 本も残らない
段階では導入しない」という判断で、production は M64×N64 / K16 のまま。本リポも同じ結論を
引き継ぎ、幾何の一般化そのものを取り込まなかった。

なお **Apple GPU 向けの GEMM 未最適化（Linux 比 31〜41 倍・タイリングが 1.21 倍しか効かない）は
未解決のまま**で、これはタイル幾何とは別の設計課題
（[2026-08-06-metal-silent-miscompute.md](2026-08-06-metal-silent-miscompute.md) §3）。

## 6. 残る候補（未着手）

参照ブランチの triage が挙げた設計候補のうち、本波で手を付けなかったもの:
adaptive norm の融合（OP-008 / F2・85 鎖）、VAE channel L2（OP-009 / F3・30 鎖・**f32 丸め
境界の保存を先に証明する必要**）、conditioner の D64 RoPE（OP-020・22 鎖）、
`PreparedExecutionPlan` によるホスト固定費の削減（HOST-006 / C1〜C4）、staged execution と
prepared cross-attention K/V（D / E）、CFG batch 化（H）。詳細は
[2026-08-06-kernel-triage/large-designs.md](2026-08-06-kernel-triage/large-designs.md)。

Metal の COMPAT-004（attention i8a8 の dp4a / emulation 不一致）と CONV-016（conv parity の
exact mismatch）は本リポの `docs/known-issues.md` 側で追跡している。
