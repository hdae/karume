> 2026-09-26 時点の実測スナップショット。Intel Arc B570（BMG G21・VRAM 9.93 GiB）/ Linux xe / Deno 2.9.6 / HEAD `4e1dc883` + ADR 0112 の実装（未コミット）。壁時計は `performance.now()`。

# anima の DiT 常駐 — B570 での利得と OOM 退避の経路

ADR [0112](../decisions/0112-anima-transformer-residency.md) の kill 判定と、決定 3（OOM で常駐 DiT を退避して段を
やり直す）を実機で踏んだ記録。台本・生ログ・JSON は `outputs/bench/karume/2026-09-26_anima-residency/`
（git 追跡外）。

## 条件

- 資産: ローカルミラー `models/karume-anima`（`karume/5`）の既定モデル `anima-turbo-v1.1`・既定席
  `f16+dit8-a8-attn8-s16`。生成は配布形の既定（8 step・CFG 1）・1024²・seed 42・台本内の固定プロンプト。
- ベンチの形: ブロック単位の ABBA（`per-stage` → `transformer` → `transformer` → `per-stage`）。1 ブロック =
  pipeline 1 本を組んで 1 + 4 回生成し dispose する。各ブロックの初回（シェーダのコンパイル。`transformer` では
  DiT の初回ロードも）を集計から外すので、n = 各 8。ブロック間は 10 s 空ける（B570 は device 破棄の解放が遅い —
  known-issues「Intel Arc B570」節）。
- 各標本が本当にそのブロックの residency で走ったことを台本が検査する（`transformer` 側は全走行で `residency`
  イベントが `retained` / `request` 1 つだけ、`per-stage` 側はイベント無し）。
- 開始時の CPU 負荷（load average）は 1.0。

## ベンチ（kill 判定）

2 回目以降の中央値（範囲は同じ 8 標本の最小〜最大）:

| residency     | n | 壁 中央値 | 壁 範囲        | DiT 段 中央値 | DiT 段 範囲    | text 2 段 中央値 | VAE 段 中央値 |
| ------------- | - | --------- | -------------- | ------------- | -------------- | ---------------- | ------------- |
| `per-stage`   | 8 | 23.49 s   | 23.21〜24.33 s | 16.77 s       | 16.54〜17.23 s | 2.05 s           | 4.62 s        |
| `transformer` | 8 | 21.05 s   | 20.81〜21.26 s | 14.15 s       | 14.15〜14.16 s | 2.23 s           | 4.63 s        |

- **利得 = 2.45 s / 生成（`per-stage` の壁の 10.4%）**。kill 線（0.5 s 未満 or 5% 未満）を超える。
- 差はほぼ DiT 段に出ている（中央値の差 2.62 s）。text 2 段は `transformer` 側が 0.17 s 長い（原因は切り分けていない）。
- 除外した初回の壁は `per-stage` 23.48 / 23.47 s、`transformer` 23.31 / 23.55 s で、両者に差は無い（常駐が効くのは
  2 回目から）。
- PNG の sha256 は 20 枚すべて 1 種類（`16f7946a…`）。参照行 `f16+dit8-a8-attn8-s16-1024` と `onEvent-1024` の
  B570 の値と一致する（常駐は数値を変えない）。

## OOM 退避 probe

`residency: "transformer"` で 1 枚生成して DiT を常駐させ、共有 device に N GiB のダミーバッファを積んでから 2 枚目を
生成する。

| ダミー | 確保した量       | 1 枚目            | 2 枚目           | 退避                                     | やり直し                                                                                        | 2 枚目の PNG sha |
| ------ | ---------------- | ----------------- | ---------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------- |
| 6 GiB  | 6.00 GiB（6 本） | 23.44 s・常駐した | 1.05 s・**失敗** | あり（text_encoder 段・開始から 1.04 s） | **失敗**: `GpuDeviceLostError`（part 1 の重みアップロード中・reason unknown / device was lost） | -                |
| 5 GiB  | 5.00 GiB（5 本） | 23.67 s・常駐した | 20.87 s・成功    | 無し                                     | -                                                                                               | 1 枚目と一致     |
| 4 GiB  | 4.00 GiB（4 本） | 23.61 s・常駐した | 20.96 s・成功    | 無し                                     | -                                                                                               | 1 枚目と一致     |

- 6 GiB では、2 枚目の text_encoder 段が OOM し、`evicted` / `out-of-memory` のイベントまでは設計どおりに出た。
  続くやり直しの text_encoder 構築で device が失われ、生成は例外で失敗した（黙っては落ちない）。
  この device lost はプロセスの panic ではなく例外として届いた。
- 5 / 4 GiB では OOM は起きず、2 枚目も常駐のまま通った。この機で退避が起きたのは 6 GiB のときだけ。

## 素の WebGPU の解放待ち probe

製品コードを使わない probe（`release-timing-probe.ts`）。待ちの形ごとに device を取り直し、1 GiB ずつ満杯まで確保 →
1 本 destroy → 待つ → 1 GiB を確保し直す → 256 MiB を `writeBuffer` して `onSubmittedWorkDone` まで通す。
結果は走行時の標準出力から写した（出力はファイルに保存していない）。

| 待ちの形                                                  | 1 GiB の再確保 | 256 MiB の書き込み |
| --------------------------------------------------------- | -------------- | ------------------ |
| 待たない                                                  | OOM            | -                  |
| `onSubmittedWorkDone` だけ                                | 成功（11 ms）  | 成功               |
| `onSubmittedWorkDone` + 空 submit + `onSubmittedWorkDone` | 成功           | 成功               |
| `onSubmittedWorkDone` + 200 ms                            | 成功           | 成功               |
| `onSubmittedWorkDone` + 1000 ms                           | 成功           | 成功               |

- 製品の解放待ち（`settleReleasedMemory` — `onSubmittedWorkDone` だけ）は、1 GiB の再確保には足りる。
- 同じ probe で、device を destroy して 500 ms 待っても、次の device で満杯までに確保できた量は
  9 → 8 → 7 → 6 GiB と減っていった（device 破棄の解放も遅い — 観測のみ）。

## 読み取り

- **採用**（opt-in のまま・既定は `per-stage`）。B570 で 2 回目以降 2.45 s / 生成（壁の 10.4%）。
- **退避 → やり直しの経路は B570 で成立していない**。退避までは動くが、やり直しの段が device lost になる。
  素の probe では `onSubmittedWorkDone` だけで再確保と書き込みが通るので、原因は解放待ちの長さではない。
  原因は**未特定**（推測: OOM を踏んだ Session 構築の後始末と次のアップロードの相互作用、または xe の
  over-commit）。調査は perf-ledger H-35、症状は known-issues「Intel Arc B570」節。
- 他機（RTX / M2）での退避経路、Chrome での利得、常駐時の text / VAE 段のピークは未計測。

## 再現

リポ直下から。GPU が学習で使われていないことを先に `outputs/diag/gpu-busy.zsh` で確かめる。

```sh
# ベンチ（kill 判定）→ results-<時刻>.json
deno run --frozen -A outputs/bench/karume/2026-09-26_anima-residency/bench.ts \
  --source models/karume-anima --resolution 1024x1024 --count 4

# OOM 退避 probe（--dummy-gib 6 / 5 / 4）→ evict-probe-<時刻>.json
deno run --frozen -A outputs/bench/karume/2026-09-26_anima-residency/evict-probe.ts \
  --source models/karume-anima --dummy-gib 6

# 素の WebGPU の解放待ち probe（引数を省くと 5 通りの待ちを全部回す）
deno run --frozen -A outputs/bench/karume/2026-09-26_anima-residency/release-timing-probe.ts
```
