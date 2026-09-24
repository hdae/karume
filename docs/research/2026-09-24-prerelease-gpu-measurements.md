# リリース前の GPU 実測 — gemma4 の run 時間の伸び / PLE メモリ内容器のフェンス本数（2026-09-24）

> 時点スナップショット（Arc B570〈xe〉/ Ryzen 5 5600 / RAM 31 GiB / Linux 7.1.8 / Deno 2.9.6〈glibc malloc〉・
> 2026-09-24）。前段の測定は [part 長と RAM ピーク](2026-09-24-part-length-ram-peak.md)（M0 / M1 / M2 の定義も
> そちら）。生データと計測スクリプトは作業用の scratchpad に置いた（git 追跡外）。本リポへの計測用の変更は無い。

## 1. gemma4 の run 時間の伸び（段 3e の M1 → M2）

前段の研究記録の読み取り 4 は「gemma4 の run が M2 で伸びたが、供給経路の遅れではない」までで、原因は
未切り分けだった。ここではその帰属を測った。

### 条件

- 資産は `models/karume-gemma4`（`karume/5`・e2b `i4-fast`・part 256）。取得元は local（`denoDirectory`）が主。
- M1 = 段 3e の候補 3 まで（`2164de24`）、M2 = 候補 3 + 候補 2(c)（`104d7cd8`）。M1 と M2 は交互に実行した。
  2 つの src の差は runtime の 3 ファイル（`executor.ts` / `session-build.ts` / `session-types.ts`）だけで、中身は
  `containerBatches` の items の lazy 化と `scopePerItem` の削除。hub と models には差が無い。
- 計測口は 3 つ。①`tools/ram-peak/measure.ts`（family = gemma4・`--max-new-tokens` 既定 8）②外付けのプローブ
  （同じ `fromPretrained` と `chat` を import し、run 中の `Deno.open`〈= host PLE の区間読み 1 回〉・
  `crypto.subtle.digest`・`/proc/self/stat` の minflt / utime / stime を数える。同じプロセスで 2 回目の chat も回す）
  ③`--v8-flags=--trace-gc` で run の窓の GC 停止時間を合計する。

### 再現（measure.ts・local・5 回・交互）

| 実装 | run ms（各回）                    | run 中央値 | load 中央値 |
| ---- | --------------------------------- | ---------: | ----------: |
| M1   | 901, 917, 881, 869, 1,025         |    **901** |       3,250 |
| M2   | 1,560, 1,412, 1,725, 1,381, 1,347 |  **1,412** |       3,246 |

差は +511 ms で、前段の研究記録の local の差（M1 873 → M2 1,367 ms）と同じ桁で再現した。

### run の中身（プローブ）

| 項目                                   | M1                      | M2                                              |
| -------------------------------------- | ----------------------- | ----------------------------------------------- |
| run0（初回 chat）ms                    | 878, 825, 843, 832, 852 | 1,357, 1,368, 1,409, 1,362, 1,322, 1,352, 1,299 |
| run1（同じプロセスの 2 回目）ms        | 324, 318, 316, 313, 319 | 331, 332, 327, 330, 329                         |
| run0 の decode 1 step（3 step 目以降） | 約 57〜65 ms            | 約 108〜124 ms                                  |
| run0 の prefill                        | 333〜367 ms             | 523〜576 ms                                     |
| run0 の `Deno.open`（区間読み）        | 28 回                   | 28 回                                           |
| run0 の `digest`                       | 28 回・939,408,400 B    | 28 回・939,408,400 B                            |
| run0 の minor page fault               | 125,749 / 117,381       | 420,963 / 380,371                               |
| run0 の sys CPU                        | 630 / 650 ms            | 1,210 / 1,140 ms                                |
| run0 の user CPU                       | 1,060 / 1,090 ms        | 1,160 / 1,100 ms                                |
| run1 の区間読み                        | 0 回                    | 0 回                                            |

- 伸びは**初回の run だけ**に出る。2 回目の run は M1 と M2 でほぼ同じ（約 +12 ms）。
- 初回 run の区間読み 28 回は、どれも **block 全体（約 32 MiB）の読みと sha256** になっている。local は未検証の
  取得元なので、`AssetReader.read` は block 全体を読んで sha256 を掛ける（runtime
  `format/container/open.ts:333-347`）。host PLE は読むたびに読み口を開き直す（`ple.ts` の MUST）ので、1 行の
  読みが毎回 block 1 本ぶんの読みと digest になる。前段の生データの local 行でも `digest.run` は 28 回・
  939,408,400 B。cold / warm は検証済みの取得元なので、run 中の digest は 0 回である。
- 作業の量（読みの回数・バイト数・digest の回数）は M1 と M2 で同じ。違うのは 1 回あたりの費用で、M2 は
  page fault が約 3.3 倍・sys CPU が約 1.8 倍になっている。

### 仮説ごとの結果

| 仮説                                                             | 結果                                                                                                                                                                                                                                 |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| (a) run の窓に host PLE の行読みが入る                           | **holds（伸びの入れ物）**。初回 run の伸びは全部この 28 回の block 読みと digest の区間にある（decode 1 step あたり +30 ms〈M1〉/ +85 ms〈M2〉・run1 では 0 回）。作業量は同じで、遅くなったのはメモリ確保の費用                     |
| (a') M2 で費用が増えた理由 = glibc malloc の動的 mmap 閾値の状態 | **holds**。閾値を固定した A/B（下の表）で、低く固定する（毎回 mmap）と M1 と M2 が並び（+17 ms）、高く固定する（heap で再利用）と差は +120 ms に縮む                                                                                 |
| (b) run 直前の GC（measure.ts の明示 GC）                        | **refuted**。measure.ts の明示 GC は component モードの Session 構築前にしか無く、family モードでは効かない。run 直前に `gc()` を足しても差は消えない（M1 1,130 / 1,154 / 1,099・M2 1,259 / 1,360 / 1,395 ms）。むしろ M1 が遅くなる |
| (c) lazy items の後に残る器の GC が run の窓に落ちる             | **holds だが寄与は小さい**。run0 の窓の GC 停止は M1 25.2 ms / M2 52.9 ms（各 1 回）。差は約 +28 ms で、+500 ms の 1 割に満たない                                                                                                    |

block 長 33,546,240 B は、glibc の動的閾値の上限 32 MiB（33,554,432 B）をわずかに下回る。そのため閾値の状態
しだいで、block 用の確保が mmap（毎回新しいページ + page fault）にも heap の再利用にもなる。

### malloc の閾値を固定した A/B（`GLIBC_TUNABLES`・交互）

| `GLIBC_TUNABLES`                                                     | 計測口        | M1 run ms     | M2 run ms     |      差 | run0 の minor fault（プローブ） |
| -------------------------------------------------------------------- | ------------- | ------------- | ------------- | ------: | ------------------------------- |
| なし（既定）                                                         | measure.ts ×5 | 中央値 901    | 中央値 1,412  |    +511 | 約 12 万 / 約 40 万             |
| `mmap_threshold=131072`（毎回 mmap）                                 | measure.ts ×3 | 中央値 1,373  | 中央値 1,390  |     +17 | 478,244 / 480,323               |
| 同上                                                                 | プローブ ×2   | 1,414 / 1,403 | 1,393 / 1,388 |     ≈ 0 | —                               |
| `mmap_threshold=33554432:trim_threshold=4294967296`（heap で再利用） | measure.ts ×3 | 中央値 877    | 中央値 997    |    +120 | 約 13.3 万 / 約 18.4 万         |
| 同上                                                                 | プローブ ×2   | 821 / 896     | 948 / 982     | 約 +100 | —                               |

補足:

- 初回 run の前に block を 8 本（約 32 MiB ずつ）読んで digest しておくと、M1 1,164 ms / M2 1,179 ms で並んだ
  （各 1 回）。run の外で大きな確保を挟むだけで M1 の速い状態は崩れる。
- cold / warm（measure.ts・3 回）では M1 → M2 の差は小さい（cold +101 ms・warm −23 ms）。run 中の digest は
  どちらも 0 回。前段の研究記録にある cold / warm の +0.14〜0.29 s は、ほとんどが M0 → M1 で出たものである
  （M0 は今回測っていない）。

### 結論

- **帰属できた**。伸びの入れ物は、run の窓に入る host PLE の block 読み（28 回・939 MB）とその sha256 である。
- M2 で伸びた原因は、この読みのバッファ確保で起きる page fault である（minor fault 約 3.3 倍・sys CPU 約
  +0.55 s）。M2 の lazy items が処理の量を増やしたわけではない。M2 の構築のあと、glibc malloc が約 32 MiB の
  確保を毎回新しい mmap で満たす状態に置かれる。
- M1 の約 0.9 s は「block 用の器を heap で再利用できる状態」に偶然置かれた結果である。run の前に `gc()` や
  大きな確保を挟むと M1 も約 1.1〜1.4 s へ崩れる。回帰ではなく、M1 のほうが測り方に対して有利な状態だった
  と読める。
- 帰属できていないのは、閾値を heap 側へ固定しても残る約 +100〜120 ms である。GC 停止の差 +28 ms はこの一部と
  重なる（残りは未切り分け）。

### 直し方の候補（未実装）

1. **根の費用を消す**: 未検証の取得元での PLE の行読みで、block の digest を 1 度だけに留める。例は、開いた
   容器が「検証済みの block 集合」を持ち、2 回目以降は区間読みにする形。初回 run の約 0.9 GB の読みと digest が
   消え、malloc の状態にも依らなくなる。代わりに、local のファイルが読みと読みの間に差し替えられる形
   （TOCTOU）の扱いを ADR で決める必要がある。
2. **計測を安定させる**: ram-peak の run 列を初回 run と 2 回目の run の 2 列に分けるか、`GLIBC_TUNABLES` を
   固定して回す。今のままでは、段どうしの比較が malloc の状態で ±0.5 s 振れる。

## 2. PLE メモリ内容器のフェンス本数

GPU 常駐席（`pleResidency: "gpu"`）の PLE は piece 1 本 = part 1 本でメモリ内容器へ渡すので、Session 構築の
フェンスが piece の本数ぶん立つ。piece を part へ束ねる価値を測った。

### 前提: 非 QAT の E2B はこの機では GPU 常駐席を使えない

非 QAT の E2B（`karume-gemma4`・i8 PLE・values 約 72 block）の PLE は、この機では GPU 常駐席を使えない。
量子化バイト列 2,348,810,240 B を単一束縛で載せる必要があり、device の `maxStorageBufferBindingSize`
2,147,483,644 / `maxBufferSize` 2,147,483,647 を超えるので fail loudly になる。そのため実測は QAT の E2B
（values 36 block）と QAT の E4B（values 22 block）で行った。資産は `models/karume-gemma4-qat`（e2b `i4-fast` /
e4b `i4`）・local。

### 計測の方法

- フェンスの回数 = PLE の Session の `buildStats.shardCount`。Session 構築は part ごとに submit を 1 回出して
  完了を待つので、part 数と等しい。
- 束ねる試作は `memory.ts` の pieces 分岐を、piece を累積 `DEFAULT_PART_BYTES`（256 MiB）まで同じ part に積む
  形にしたもの（scale は piece 1 の part に数える — 規則③は不変）。
- ホスト RAM は PLE 構築の窓で `Deno.memoryUsage()` を 10 ms ごとに取った。VRAM は同じ間隔で
  `/proc/self/fdinfo/<renderD の fd>` の `drm-total-vram0` を取った（`drm-total-gtt` は全区間 0）。
- 各構成 3 回・交互。chat 8 token の出力文字列は束ねる前と後で一致した。

### 数値（中央値・3 回）

| 系列                       | 形     | フェンス本数 | uploadFenceMs | フェンス 1 回 | writeBufferIssueMs | shardWaitMs | PLE 構築 ms | fromPretrained 全体 ms | VRAM 最大（PLE 窓） |
| -------------------------- | ------ | -----------: | ------------: | ------------: | -----------------: | ----------: | ----------: | ---------------------: | ------------------: |
| QAT E2B（values 36 block） | 現状   |           37 |           485 |       13.1 ms |                128 |       2,128 |       2,849 |                  5,002 |           1,381 MiB |
| 〃                         | 束ねる |            6 |           118 |       19.7 ms |                289 |       2,100 |       2,616 |                  4,559 |           1,573 MiB |
| QAT E4B（values 22 block） | 現状   |           23 |           317 |       13.8 ms |                101 |       1,543 |       2,083 |                  6,070 |             933 MiB |
| 〃                         | 束ねる |            4 |            72 |       18.0 ms |                160 |       1,257 |       1,598 |                  5,899 |           1,125 MiB |

ホストの external 最大 / rss 最大は E2B 208 → 176 MiB / 566 → 569 MiB、E4B 190 → 190 MiB / 577 → 577 MiB で、
ほぼ動かない。E4B の shardWaitMs と PLE 構築の壁時計は 1 回目（現状 1,959 / 2,544 ms）が page cache の冷えで
大きく、ばらつきの主はそこにある。

### 読み取り

- フェンス 1 回 13.1〜13.8 ms は、ADR [0108](../decisions/0108-container-format.md) 決定 9 の 13.0 ms / 回と合う。
- 束ねるとフェンスの合計は E2B で −367 ms、E4B で −245 ms 減る。ただし writeBuffer の発行が +161 ms / +59 ms
  増える（staging が part ぶん溜まる間の複製の費用という読みは**推測**・機序は未確認）。差し引きした上げの
  費用（フェンス + 発行）は E2B 613 → 407 ms（**−206 ms**）、E4B 418 → 232 ms（**−186 ms**）。
- PLE の構築全体では E2B 2,849 → 2,616 ms（−233）、E4B 2,083 → 1,598 ms（−485 — shardWait の揺れを含む）。
  構築の主な費用はフェンスではなく shardWaitMs（local の block 読みと sha256 で約 1.3〜2.1 s）で、1 の直し方 1
  と同じ根である。
- 代償は構築時の VRAM の一時的な上乗せで、E2B / E4B とも **+192 MiB**。staging の上限が piece 1 本（32 MiB）
  から part 1 本（256 MiB）へ上がったぶんと読む。この機の xe では staging が gtt ではなく vram0 に計上されていた
  （機序は未確認）。ホストの external / rss が動かないのは、M2 の形では item を 1 本ずつ上げて手放すので、
  part を大きくしても JS 側は block 1 本のままだからである。
- **推測**: 束縛上限の大きい device での非 QAT E2B（72 block）は、現状約 73 × 13 ms ≈ 0.95 s、束ねると約
  10 part × 約 20 ms ≈ 0.2 s に発行の増分（QAT E2B の +161 ms を量に比例させて約 +0.3 s）が乗り、差し引き
  約 −0.4〜−0.5 s。この機では測れない。

### 結論

- **束ねる価値は小さいが正**（QAT E2B / E4B で PLE 構築 −0.19〜−0.21 s・fromPretrained 全体の約 3〜4%）。
  代償は構築時の VRAM +192 MiB。以前の見積り「約 0.94 s」は GPU 常駐席を使えない非 QAT E2B の数だった。
- 束ねる場合の実装の規模:
  1. `memory.ts` の pieces 分岐の part 割りを、累積が `DEFAULT_PART_BYTES` まで同じ part に積む形へ変える
     （約 15 行・試作と同じ形）。
  2. 契約と doc: `memory.ts` の `WHOLE_PART` の doc（「piece は 1 本 = 1 part = フェンス 1 回」）、ADR 0108 の
     追記（メモリ内容器の part の割り方・staging の上限が piece → part）、`ple-gpu.ts` のモジュール doc。
  3. テスト: `packages/runtime/tests/memory_container_test.ts:378`（「piece は 2 の次から 1 本 1 part」を直接
     assert している）を新しい割り方の assert に置き換え、GPU の e2e で出力の一致を確かめる。

## 3. 未確認

- 1: M2 の構築のあとで glibc の閾値が「毎回 mmap」側に残る機序（どの free が閾値を動かすか）。**推測**の段階で、
  malloc の内部は観測していない。
- 1: 閾値を heap 側へ固定しても残る約 +100〜120 ms の内訳（GC 停止の差 +28 ms を除いた残り）。
- 1: cold / warm の M0 → M1 の伸び（M0 は測っていない）。trace-gc の集計は 1 回ずつで、中央値ではない。
- 2: VRAM の値は fdinfo の 10 ms 標本化の最大値で、短い山を取りこぼしうる。staging が vram0 に計上される機序は
  未確認。
- 2: writeBufferIssueMs が束ねると増える機序（staging の確保の粒度か、複製先のメモリの種別か）。
- 2: 非 QAT E2B（72 block）の数値は全部**推測**（この機では fail loudly になり測れない）。ブラウザ（Chrome）は
  測っていない。
