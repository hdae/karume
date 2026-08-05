# タスク #20 recon — 非 GPU 固定費 695〜704ms/step の帰属と修正候補（2026-08-04）

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。

> 読み取り専用レッグの成果物。**リポジトリのファイルは 1 バイトも変更していない**
> （`git status --porcelain` は開始時と同じ `?? .claude/scheduled_tasks.lock` のみ）。
> 計測物は全て scratchpad に置いた。数値は全て本日の実出力からの転記で、私が導いた値は
> 「**試算**」、機序からの推論は「**推測**」と明示する。
> プロトタイプ（参考元の先行実験プロジェクト）は参照していない。

---

## 0. 要旨

1. **「非 GPU 固定費」は 2 つの別物の和だった**。1024px step 10 の 717.8ms は
   **① GPU 時間診断の装置代 375.2ms（52.3%）**＋**② 真のホストエンコード代 308.5ms（43.0%）**
   ＋ ③ submit 間隙 34.0ms（4.7%）に分かれる。①は `gpuTiming` **自動有効**（`acquireGpu({})`
   = 既定）で発動する「1 dispatch = 1 pass」（ADR 0021）の代価で、**解像度非依存**
   （512px でも 369.8ms）。
2. **①は step 番号に単調増加する**（1024px: step 2 で 470.4ms → step 10 で 705.6ms）。
   増加分は全て `encoder.finish()` で、単価は **1 pass あたり 40µs（step 2）→ 115µs（step 10）**。
   単離実験で機序を確定: `finish` の代価は **pass 数に比例**し、かつ **その device で
   これまでに作られたバッファ総数に比例**して単価が上がる（0 本 30.7µs/pass →
   25,000 本 129.7µs/pass）。Karume は **1 step で 3,356 本の GPUBuffer を作って壊す**ので、
   step を重ねるほど①が太る。**gpuTiming を切ると増加は消える**（1,654〜1,690ms で平坦）。
3. **本当の根治点はどちらでもなく「CPU と GPU が重なっていないこと」**。
   `SubmitScheduler.#submitChunk` の `void queue.onSubmittedWorkDone().then(...)`
   （submit.ts:441）は、Deno/wgpu では**同期部分がその時点までの GPU 完了までブロックする**
   （実測: 1 呼び出しあたり 34.8ms・1 step 合計 1,672.7ms ≒ GPU pass 和 1,638.7ms）。
   結果、壁時計 = GPU + ホストの**厳密な加算**になっている。素の WebGPU 単離実験で
   「呼ぶと wall=gpu+host（47.2ms）/ 呼ばないと wall=max(gpu,host)（30.6ms）」を確認。
4. **実機での擬似実験（onSubmittedWorkDone を非ブロッキング化）で、1024px の非 GPU は
   705.6 → 163.0ms（−77%）、512px は 646.8 → 269.1ms（−58%）**。壁時計は 512px で
   1,050 → 632ms/step。**①も②もほぼ丸ごと GPU の裏に隠れる**。
5. 推奨は **案 1（submit ごとの `onSubmittedWorkDone` を廃し、計測を flush 1 回へ集約）**。
   次点は **案 3（`gpuTiming` の既定を off へ）= 工数 S で −370〜375ms/step**、
   **案 2（params バッファ / bind group のキャッシュ）= −139〜191ms/step + ①の単価成長の停止**。
6. 旧候補の**否定**: `errorScope` は 1 run 4 呼び出し・計 0.03ms（**無罪**）。GC は
   `--trace-gc` で 1 step あたり Scavenge 2〜4 回・各 0.2〜0.9ms、mutator utilization 0.999
   （**無罪**）。`await` 連鎖は残差 JS 全体でも 81.4ms（非 GPU の 11.3%）で**主因ではない**。

---

## 1. 方法

### 1.1 環境

RTX 3080 Ti（12,288MiB）/ Deno 2.9.4 / wgpu-Vulkan。開始時 GPU 使用 33MiB・SM 210MHz・41℃
（本レッグが GPU を専有）。`performance.now()` の分解能は実測で 0.4µs 級。

### 1.2 計測物（全て scratchpad・リポジトリ外）

| ファイル          | 役割                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hostprobe.ts`    | `GPUDevice` / `GPUQueue` / `GPUCommandEncoder` / `GPUComputePassEncoder` / `GPUComputePipeline` / `GPUBuffer` / `GPUQuerySet` の prototype を包み、呼び出し数と**同期実行 ms** を集計。async 系は `[sync]`（返るまで）と `[resolve]`（解決まで）の 2 本 |
| `probe_run.ts`    | `mod.ts` を直接使い DiT 段だけを再現するハーネス（入力は形だけ同じダミー値）。`--mode timed/count/off` `--timing on/off` `--steps N` `--res 512/1024` `--euler` `--stub-onswd`                                                                          |
| `micro_onswd.ts`  | 素の WebGPU で「submit ごとに `onSubmittedWorkDone` を呼ぶ / 呼ばない」を A/B                                                                                                                                                                           |
| `micro_finish.ts` | 素の WebGPU で `encoder.finish()` の代価が pass 数 / 累積バッファ数のどちらに比例するかを単離                                                                                                                                                           |

実行例（全て `deno run -A --config /home/developer/workspace/karume/deno.json`）:

```sh
deno run -A --config .../deno.json probe_run.ts --mode timed --timing on --steps 10 --res 1024 --euler
deno run -A --config .../deno.json probe_run.ts --mode off   --timing on --steps 10 --res 1024 --euler --stub-onswd
```

`--res 1024` は `models/anima-turbo-i8-1024/transformer`、`--res 512` は
`models/anima-turbo-i8/transformer`。`linearCompute: "i8a8"` / `attentionCompute: "i8a8"` を
既定にしてあり、**wave1-bench.sh の運用形（`--turbo --dit i8 --resolution 1024
--linear-compute i8a8 --attention-compute i8a8`）と同じ構成**。

### 1.3 ハーネスの忠実性（アンカー）

実デモを 1 回まわして本 recon のハーネスと突き合わせた（`--out` は scratchpad へ逃がしたので
`models/` も無変更）:

```
[dit]  step 10/10 σ=0.2500 2041ms / submit 40 dispatch 3311 chunk 1024
[time] transformer GPU 計 1348ms   → 非 GPU 693ms
```

→ **正本 doc の 695ms を独立再現**。ハーネス（mode=off / timing=on / 10 step）の同条件は
step 10 で `runMs 2383.29 / gpuMs 1677.68 / 非 GPU 705.61`（GPU 側は熱状態で振れるが
**非 GPU は 693〜723ms で 4 回とも一致**）。

### 1.4 計測ラッパ自体の歪み

`--mode timed`（全ラップ + 2 × `performance.now()`）と `--mode off`（無加工）の壁時計:

| 構成               | mode=off step2 / step10 |          mode=timed step2 / step10 |
| ------------------ | ----------------------: | ---------------------------------: |
| 1024px・timing off |     1,677.32 / 1,689.65 |                1,669.59 / 1,688.41 |
| 1024px・timing on  |     1,867.17 / 2,383.29 | 1,942.33 / 2,356.44（別 run の熱） |

→ **ラッパ由来の歪みは ±10ms/step 未満**（非 GPU 700ms に対し 1.5% 未満）で、結論を動かさない。
26,700〜34,000 呼び出しを包んでこの安さなのは、包んだ呼び出しの大半が µs 級の実仕事を持つため。

---

## 2. コード recon — 1 dispatch あたりホスト側で何が走るか

1 ノード = 1 dispatch（attention / linear i8a8 等は 1 ノード = 3〜7 dispatch）。
`src/runtime/executor.ts:731` の
`for (const step of plan.nodes) await this.#encodeNode(step, env, arena);` が全体のループで、
1 dispatch ごとに次が走る（代表として elementwise 経路 `executor.ts:985-1024`。
**`createBindGroup` は executor 内に 30 箇所**あり、どれも同じ形）:

| # | 仕事                                 | file:line                                                  | 備考                                                                                          |
| - | ------------------------------------ | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| ① | `await cache.get(key, wgsl)`         | `executor.ts:999` / `gpu/pipeline-cache.ts:43-52`          | ヒット時も `async` なので **microtask 1 回**。WGSL 文字列の生成は各 `*Wgsl(spec)` 側          |
| ② | params の `Uint32Array` 組み立て     | `executor.ts:1001-1009`                                    | shape / stride / スカラ attrs。JS 残差に入る                                                  |
| ③ | **params バッファの `createBuffer`** | `executor.ts:2333-2337` → `gpu/arena.ts:123-133`           | `allocHostWritten` は**プール対象外**（writeBuffer 追い越しの不変条件）→ **毎 dispatch 新品** |
| ④ | **`queue.writeBuffer(params)`**      | `executor.ts:2335`                                         | 同上                                                                                          |
| ⑤ | `pipeline.getBindGroupLayout(0)`     | `executor.ts:1011`                                         | 毎回呼ぶ（キャッシュしていない）                                                              |
| ⑥ | **`device.createBindGroup`**         | `executor.ts:1010-1017`                                    | entries 配列も毎回新規生成                                                                    |
| ⑦ | `gridStrideWorkgroups`               | `executor.ts:1018-1022`                                    | 純計算                                                                                        |
| ⑧ | `scheduler.dispatch(...)`            | `executor.ts:1023` → `gpu/submit.ts:297-313`               | `#pending` に push。**予算 or 上限に達したらその場で `#submitChunk()`**                       |
| ⑨ | 出力バッファの確保                   | `executor.ts:~950`（各 encode の手前）→ `arena.ts:101-117` | プール適格。実測は 3,877 回中 3,835 回が再利用・新規 42 回                                    |
| ⑩ | ノード境界の `arena.release`         | `executor.ts:974-982`                                      | 参照計数（Map 操作）                                                                          |

チャンク境界（`#submitChunk` — `gpu/submit.ts:417-462`）で追加に走るもの:

| 仕事                                                                             | file:line           | 計測が**無効**なとき | 計測が**有効**なとき（ADR 0021）                  |
| -------------------------------------------------------------------------------- | ------------------- | -------------------- | ------------------------------------------------- |
| `createCommandEncoder`                                                           | `submit.ts:425`     | 1 回/チャンク        | 同左                                              |
| `beginComputePass` / `pass.end`                                                  | `submit.ts:429-435` | **1 回/チャンク**    | **1 回/dispatch**（`submit.ts:471-503`）          |
| `createQuerySet` + `createBuffer` × 2 + `resolveQuerySet` + `copyBufferToBuffer` | `submit.ts:482-505` | 無し                 | 1 組/チャンク                                     |
| `encoder.finish()`                                                               | `submit.ts:438`     | 1 回/チャンク        | 同左（ただし**中身の pass 数が 3,311 に膨らむ**） |
| `queue.submit()`                                                                 | `submit.ts:438`     | 1 回/チャンク        | 同左                                              |
| **`queue.onSubmittedWorkDone()`（fire-and-forget）**                             | **`submit.ts:441`** | **1 回/チャンク**    | 同左                                              |

run 境界（`executor.ts:717-760`）:

- `pushFailureScopes` / `popFailureScopes` は **run に 2 本 ×（本体 + readback）= 4 回**
  （`device.ts:658-687`）。`withScopeLock` は run 全体を直列化（`device.ts:558`）。
- `flush()`（`submit.ts:342-349`）= 最後のチャンクを submit → `onSubmittedWorkDone` を await →
  `#collectTimings()`（`submit.ts:522-541`）で全チャンクの `mapAsync` を待って集計。
- `#readOutputs`（`executor.ts:2340-`）= staging へ copy → 別 submit → `mapAsync`。
- `arena.destroy()`（`arena.ts:250-267`）で **run が作った全バッファ（3,356 本）を `destroy()`**。

`gpuTiming` の既定は **`undefined` = 自動判定 = アダプタが持てば有効**
（`device.ts:160-174` の `planTimestampFeature`、doc は `device.ts:355-370`）。
デモは `acquireGpu(needsShaderF16 ? { shaderF16: true } : {})`（`examples/anima/main.ts:583`）
なので **常に有効側に落ちる**。

---

## 3. 実測帰属

### 3.1 1024px・step 10（`--mode timed --timing on --steps 10 --euler`・`timedfull1.log`）

`runMs 2,356.44` / `gpuMs（pass 和）1,638.68` / **非 GPU 717.76** / dispatch 3,311 /
チャンク 47 / arena alloc 3,356・reuse 3,835。

| 区分                                                         |         ms |  非 GPU 比 | 内訳の根拠                                                                                             |
| ------------------------------------------------------------ | ---------: | ---------: | ------------------------------------------------------------------------------------------------------ |
| **① `encoder.finish()`**                                     | **378.94** |  **52.8%** | n=47・8,062µs/call = **114.7µs/pass**（3,311 pass ÷ 47 チャンク）                                      |
| ② JS グルー（残差 — params 組み立て・Map・microtask・GC 等） |      81.44 |      11.3% | 総和 2,275.00 に対する `runMs` の残り                                                                  |
| ③ `device.createBuffer`                                      |      75.37 |      10.5% | n=**3,448**（params 3,311 + 入力 3 + 新規 storage 42 + query 92）                                      |
| ④ `device.createBindGroup`                                   |      52.16 |       7.3% | n=3,311・15.75µs/call                                                                                  |
| ⑤ `queue.writeBuffer`                                        |      44.61 |       6.2% | n=3,314・13.46µs/call                                                                                  |
| ⑥ `onSubmittedWorkDone[sync]` の GPU pass 和 超過分          |      34.02 |       4.7% | 1,672.70 − 1,638.68（= submit 間隙・pass 外のドライバ時間）                                            |
| ⑦ `buffer.destroy`                                           |      25.63 |       3.6% | n=3,448                                                                                                |
| ⑧ `mapAsync` 待ち（timestamp 回収）                          |      10.56 |       1.5% | 47 本同時発行なので **1 本ぶんの実効値**（496.55 ÷ 47）                                                |
| ⑨ その他の同期呼び出し 16 種                                 |      15.03 |       2.1% | submit 4.12 / beginComputePass 3.14 / getBindGroupLayout 2.84 / pass.* 2.09 / createQuerySet 0.76 / 他 |
| **計**                                                       | **717.76** | **100.0%** | **表と非 GPU の残差 0.00ms**                                                                           |

> **整合の読み方**: ②は残差そのものなので合計一致は構成上の恒真。**正直な検算は
> 「名前の付いた項の総和 2,275.00ms が `runMs` 2,356.44ms の 96.5% を占める」**の方で、
> 未帰属は 3.5%（= ②）に収まっている。
>
> `onSubmittedWorkDone[sync]` の 1,672.70ms は**ほぼ全部が GPU 待ち**なので非 GPU には
> 超過分⑥だけを計上した（この 1 本が本 recon の主題 — §4.1）。

### 3.2 解像度・計測有無のクロス（全て step 10・`--euler`）

| 構成                   | 壁 ms/step | GPU pass 和 | 非 GPU |   `finish` | `createBuffer` | `createBindGroup` | `writeBuffer` | JS 残差 |
| ---------------------- | ---------: | ----------: | -----: | ---------: | -------------: | ----------------: | ------------: | ------: |
| 1024px・timing **on**  |   2,356.44 |    1,638.68 | 717.76 | **378.94** |          75.37 |             52.16 |         44.61 |   81.44 |
| 1024px・timing **off** |   1,688.41 |         n/a |    n/a |  **10.78** |          80.51 |             50.28 |         48.81 |   66.45 |
| 512px・timing **on**   |   1,050.42 |      403.63 | 646.79 | **371.72** |          52.73 |             49.04 |         42.39 |   75.29 |
| 512px・timing **off**  |     667.72 |         n/a |    n/a |   **7.38** |          51.36 |             47.66 |         43.32 |   63.41 |

**読み取り**:

- **① 計測装置代 = timing on と off の「pass 由来の項」の差**
  （`finish` の差 + `beginComputePass`/`pass.end` の差 + querySet 資源 + query バッファ）
  → 1024px **375.2ms**（368.16 + 3.09 + 1.00 + 0.76 + 0.10 + createBuffer 92 本ぶん ≈ 2.0）・
  512px **369.8ms**（364.34 + 2.94 + 1.00 + 0.70 + 0.10 + 0.7）= **解像度非依存**。
  **検算**: 非 GPU 717.76 − ① 375.2 − ③ 34.02 = **308.5**（1024px）に対し、
  timing off から直接測った②が **301.9**（差 6.6ms = 2%）。512px も 249.7 対 237.9（差 11.8ms）。
- **② 真のホスト代 = 壁 − `onSubmittedWorkDone[sync]`**（timing off で読む）
  → 1024px **301.9ms**（1,688.41 − 1,386.51）・512px **237.9ms**（667.72 − 429.79）。
  1024px の方が 64ms 高いのは `createBuffer` + `destroy` の差（105.4 対 59.3ms）= バッファが
  大きいぶん確保/破棄が高い。**dispatch 数は同じ 3,311 なので、残りの項はほぼ完全に一致**。
- **③ 1 dispatch あたりに正規化**すると、②は **91µs/dispatch（1024px）/ 72µs/dispatch（512px）**。

### 3.3 「非 GPU 固定費」は step 番号に単調増加する（= 固定費ではない）

| step | 1024px timing on 壁 | GPU pass 和 |     非 GPU | 1024px timing **off** 壁 |
| ---: | ------------------: | ----------: | ---------: | -----------------------: |
|    2 |            1,867.17 |    1,396.81 | **470.36** |                 1,677.32 |
|   10 |            2,383.29 |    1,677.68 | **705.61** |                 1,689.65 |

`--timing off` では 1,654〜1,690ms で**完全に平坦**。増加分は全て `encoder.finish` の単価
（step 2 で 3,169µs/call ÷ ~80 pass = **40µs/pass** → step 10 で 8,062µs ÷ 70 pass =
**115µs/pass**）。**熱ではない**（熱なら timing off でも伸びる）。

> **ADR 0030 の「~350ms/step」との 2 倍差の説明（推測を含む）**: 非 GPU は step 2 で 470ms・
> step 10 で 706ms（1024px）、512px でも 432 → 647ms と**どの step を読むかで 1.5 倍動く**。
> さらに①を含むか否かで 2 倍動く。ADR 0030 の値は②+③の帯（1024px 336ms / 512px 265ms）に
> 近く、**「装置代を含まない読み」か「若い step の読み」のいずれか**だったと推測する。
> どちらにせよ「非 GPU は固定費」という言い方自体が不正確で、**「dispatch 数 × 単価 +
> 累積バッファ数に比例する装置代」**が正しい。

### 3.4 旧候補の判定（正本 doc §10-2 の列挙に対する回答）

| 候補                         | 判定                                                                                                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| bind group 生成              | **有罪（小）** 52.16ms/step = 非 GPU の 7.3%                                                                                                                             |
| params バッファ              | **有罪（中）** `createBuffer` 75.37 + `writeBuffer` 44.61 + `destroy` 25.63 のうち params 由来が **96%（3,311/3,448）**≒ **139ms/step**。加えて①の単価成長の燃料（§4.2） |
| `await` 連鎖                 | **主因ではない** 残差 JS 全体でも 81.44ms（11.3%）。microtask は 1 dispatch 1 回                                                                                         |
| submit 分割の適応制御        | **直接は無罪**（`submit` 4.12ms + `createCommandEncoder` 0.09ms）。**ただし `onSubmittedWorkDone` の呼び出し点を作っている**（§4.1）                                     |
| errorScope                   | **無罪** push/pop 各 4 回・計 **0.03ms/run**                                                                                                                             |
| GC                           | **無罪** `--trace-gc` 実測で 1 step あたり Scavenge 2〜4 回・各 0.2〜0.9ms、mutator utilization **0.999**（Mark-Compact は 4 step で 2 回・各 0.7ms）                    |
| **（新）GPU 時間診断の装置** | **最大の有罪** 370〜375ms/step・**解像度非依存**・step で増加                                                                                                            |
| **（新）CPU/GPU の非重複**   | **構造的な根本原因** — §4.1                                                                                                                                              |

---

## 4. 機序の単離（なぜそうなるか）

### 4.1 `queue.onSubmittedWorkDone()` は Deno では同期ブロックする

`submit.ts:441` の fire-and-forget 呼び出しの**同期部分**（promise が返るまで）が
**1 呼び出し 34.8ms・1 step 合計 1,672.70ms**（1024px step 10・n=48）。GPU pass 和 1,638.68ms
とほぼ一致するので、「その時点までに submit した全作業の完了までブロックしている」と読める。

素の WebGPU で単離（`micro_onswd.ts` — 40 チャンク、各チャンク = ホスト API 80 回 + 1 dispatch）:

| round | GPU のみ | ホストのみ | **A: submit ごとに onSWD** | A の onSWD sync 合計 | **B: 呼ばない** |
| ----: | -------: | ---------: | -------------------------: | -------------------: | --------------: |
|     0 |     33.6 |       17.6 |                   **50.7** |                 23.0 |        **33.7** |
|     1 |     33.4 |       15.2 |                   **47.2** |                 19.7 |        **30.6** |
|     2 |     30.2 |       14.9 |                   **47.3** |                 19.7 |        **30.6** |

**A ≒ gpu + host（48.6 対 47.2）/ B ≒ max(gpu, host)（33.4 対 30.6）**。
→ 現行実装は CPU エンコードと GPU 実行を**完全に直列化**している。これが
「非 GPU が GPU 時間にそのまま足し算で乗る」という観測の正体。

> **注意（移植性）**: これは Deno + wgpu 実装の性質で、ブラウザの `onSubmittedWorkDone` は
> 非ブロッキング。**したがって案 1 はブラウザでは無害（呼ばなくなるだけ）で、Deno で効く**。
> なお Karume 側にこれを回避する API は無い（Web 標準 API のみ — ADR 0002）。

### 4.2 `encoder.finish()` は「pass 数 × 累積バッファ数」で高くなる

`micro_finish.ts`（100 pass / 400 pass の command buffer を、バッファを 5,000 本ずつ増やしながら
`finish` する。stage 4-5 は**作って即 destroy**＝生存させない）:

| stage | 生存バッファ | 使い捨て済み | finish(100 pass) | finish(400 pass) | **µs/pass(100)** | **µs/pass(400)** |
| ----: | -----------: | -----------: | ---------------: | ---------------: | ---------------: | ---------------: |
|     0 |            0 |            0 |            3.068 |            9.276 |            30.68 |            23.19 |
|     1 |        5,000 |            0 |            3.067 |           11.801 |            30.67 |            29.50 |
|     2 |       10,000 |            0 |            5.557 |           21.814 |            55.57 |            54.53 |
|     3 |       15,000 |            0 |            8.068 |           31.654 |            80.68 |            79.13 |
|     4 |       15,000 |        5,000 |           10.497 |           41.303 |           104.97 |           103.26 |
|     5 |       15,000 |       10,000 |           12.969 |           50.748 |           129.69 |           126.87 |

- **pass 数に対して厳密に線形**（100 pass と 400 pass の µs/pass が全 stage で一致）。
- **`destroy()` しても単価は下がらない**（stage 3→5）→ 効くのは**生存数ではなく累積生成数**。
  約 **+5µs/pass / 1,000 バッファ**。**推測**: wgpu 側のリソーストラッカーが id 空間の大きさで
  確保されるため（Karume からは観測できないので機序は推測、**相関は実測**）。
- Karume は 1 step で **3,356 本**を作って壊すので、step を重ねるほど①が太る（§3.3 の実測と整合）。
  実測の伸び（8 step で +75µs/pass = 2.8µs/1,000 本）は micro（4.95µs/1,000 本）と同オーダー。

### 4.3 なぜ①が 380ms にもなるか

`gpuTiming` 有効時は **1 dispatch = 1 pass**（`submit.ts:471-503`）なので、1 step の pass 数は
チャンク数（33〜47）ではなく **dispatch 数 3,311** になる。
**3,311 pass × 114.7µs = 379.8ms** ≒ 実測 378.94ms。512px も **3,311 × 112.3µs = 371.8ms**
≒ 実測 371.72ms。**解像度非依存であることの説明もこれで尽きる**（pass 数が同じだから）。

### 4.4 擬似実験 — 案 1 を入れたら何が起きるか（実機・Karume 本体）

`GPUQueue.prototype.onSubmittedWorkDone` を**非ブロッキングな promise**に差し替えて
（`probe_run.ts --stub-onswd`）同じ 10 step を回した。実際の GPU 同期点は `#readOutputs` の
`mapAsync` が担うので run の出力自体は正しいまま（出力は検証していない — 時間のみの実験）。

| 構成                  | 壁 ms/step | GPU pass 和 | **非 GPU** |
| --------------------- | ---------: | ----------: | ---------: |
| 1024px 現行（対照）   |   2,383.29 |    1,677.68 | **705.61** |
| 1024px stub（1 回目） |   1,619.80 |    1,456.82 | **162.98** |
| 1024px stub（2 回目） |   1,619.15 |    1,458.38 | **160.77** |
| 512px 現行（対照）    |   1,050.42 |      403.63 | **646.79** |
| 512px stub            |     632.32 |      363.18 | **269.13** |

- **1024px: 非 GPU 705.6 → 161.9ms（2 回平均・−77%）**。GPU（1,457ms）> ホスト（約 690ms）
  なので、①も②もほぼ丸ごと裏に隠れた。残る 162ms は最初のチャンクを積むまでの立ち上がりと
  readback・run 境界（重ねられない部分）。
- **512px: 非 GPU 646.8 → 269.1ms・壁 1,050 → 632ms/step（−40%）**。こちらは
  ホスト（約 620ms）> GPU（約 380ms）なので **wall ≒ ホスト代**に張り付く（モデル
  `wall ≈ max(host, GPU)` の予測どおり）。
- **開示（上振れ要因）**: stub は適応制御の実測を壊すので chunk が上限 1,024 まで育ち、
  submit は 33〜47/step → 3〜11/step に減っている。ただし①の主項 `finish` は**チャンク数では
  なく pass 数**で決まる（§4.2）ので、チャンク数を現行のまま保っても隠れる側の結論は変わらない
  （**推測**: 差は数十 ms 級）。

---

## 5. 修正候補

### 案 1（推奨）— submit ごとの `onSubmittedWorkDone` を廃し、計測を flush 1 回へ集約

**形**: `SubmitScheduler.#submitChunk`（`submit.ts:437-461`）から fire-and-forget の
`onSubmittedWorkDone().then(...)` を外す。適応制御の観測点を `flush()`（`submit.ts:342-349`）
1 回に移し、**窓 = 「窓の最初の submit 時刻 → flush 完了時刻」・仕事量 = 窓の合計 workgroup 数**
で `msPerWorkgroup` を更新する。

- **期待利得（実測ベース）**: 1024px **−543ms/step**（非 GPU 705.6 → 162.0）、
  512px **−378ms/step**（646.8 → 269.1・壁 −418ms/step）。1024px turbo 10 step の壁時計では
  **−5.4s 級**（試算: 543ms × 10）。
- **工数**: **M**（submit.ts の計測経路の再設計 + 既存テストの更新）。
- **触るファイル**: `src/gpu/submit.ts`（`#submitChunk` / `flush` / `#closeMeasurementWindow` /
  `SubmitStats.measuredMs` の意味）、`tests/` の submit 系（`measuredMs` の粒度を見ているもの）。
- **リスクと既存不変条件の判定**:
  - **ADR 0004 適応制御**: 不変条件 1（実測 0 は情報なし）は**保てる** — 窓の合計が 0 なら更新
    しない、をそのまま移植。不変条件 3（workgroup 単位のフィードフォワード）も**保てる**。
    不変条件 2（積む前に判定）は**無変更**。ただし**新しい窓値はホスト側の詰まりも含む上限**に
    なる → `msPerWorkgroup` を過大評価する向き＝**チャンクが小さくなる向き**＝ TDR に対して
    安全側。**要注意**: 512px のようにホスト律速だと過大評価が大きく、チャンクが必要以上に
    細るおそれ（`minChunkSize` と `maxChunkSize` が歯止め）。**GPU timestamp が有効なときは
    そちらを推定源にする**という改良余地あり（別案）。
  - **「1 submit 全積み」の罠（Pitfalls）**: 計測の粒度が粗くなるので、**「実測 0 を成長の
    根拠にしない」の門は必ず残す**こと。ここが唯一の再発点。
  - **errorScope 直列化（`GpuContext.withScopeLock`）**: **無関係** — GPU 操作の発行位置は
    変わらず、削るのは待ちだけ。`.then` コールバックが JS カウンタしか触らない点も同じ。
  - **flush-before-destroy**: **無変更** — `flush()` は引き続き `onSubmittedWorkDone` を await
    する（`arena.destroy()` はその後）。
  - **codegen 決定性**: **無関係**（WGSL もキーも触らない）。
  - **移植性**: ブラウザでは元々非ブロッキングなので**挙動は変わらない**（Deno でだけ効く）。

### 案 2 — params バッファと bind group のキャッシュ

**形**: (a) params は「ノード + 解決済み shape」で決まる不変値なので、**Session 常駐の
キャッシュ**にして run をまたいで使い回す（初回だけ `createBuffer` + `writeBuffer`）。
(b) さらに bind group もキャッシュするには、入出力バッファが run 間で同一である必要があるので
**`RunArena` を Session 常駐にする**（同じ shape なら確保列が決定的なので同じ割り当てになる）。

- **期待利得（実測ベース）**: (a) だけで 1024px **−139ms/step**（`createBuffer` 75.37 ×
  3,311/3,448 + `writeBuffer` 44.61 + `destroy` 25.63 × 3,311/3,448）、512px **−103ms/step**。
  (b) を足すと **+52ms**（`createBindGroup`）。
  加えて**①の単価成長が止まる**（§4.2）— gpuTiming 有効時は step 10 で `finish` 378.94ms の
  うち成長ぶん **≈247ms**（試算: (115−40)/115 × 378.94）が消える。
- **工数**: (a) **M** / (a+b) **L**。
- **触るファイル**: `src/runtime/executor.ts`（`#writeParams` と 30 箇所の bind group 生成）、
  `src/gpu/arena.ts`（(b) のとき寿命の変更）、`tests/gpu_arena_test.ts` 系。
- **リスクと既存不変条件の判定**:
  - **ADR 0004「`writeBuffer` で書くバッファはプール外」**（`arena.ts:11-13,119-133`）:
    (a) は**両立する** — キャッシュした params は**二度と書き換えない**ので、
    「未 submit の先行エンコードを writeBuffer が追い越す」ハザードが原理的に起きない。
    **MUST**: キャッシュキーに**解決済み shape / スカラ attrs の実値**を含める（記号次元の
    セッションで値が変わる。ここを外すと沈黙誤値）。
  - **full-write 不変条件（ADR 0014）**: **無関係**（params は出力ストレージではない）。
  - **flush-before-destroy**: (b) はアリーナの寿命が run から Session へ伸びるので、
    **`Session.dispose()` 側に flush-before-destroy を持たせ直す**必要がある（現在は run 末尾）。
    ここが (b) の主リスク。
  - **VRAM**: (b) は transient（1024px DiT で 712.5MiB — ADR 0031）が run 間も常駐する。
    デモの段構成（DiT を dispose してから VAE をロード）とは両立するが、
    **known-issues の GPUBuffer 天井 7,280MiB の見積もりは取り直しになる**。
  - **codegen 決定性 / errorScope**: **無関係**。
- **順序の注意**: **案 1 を先に入れると 1024px では案 2 の利得はほぼ観測できなくなる**
  （GPU の裏に隠れる）。効くのは**ホスト律速側（512px 以下・小さいグラフ・SBV2/DeBERTa）**。

### 案 3 — `gpuTiming` の既定を「自動有効」から「無効」へ（または 1 dispatch = 1 pass を別ノブへ）

**形**: `planTimestampFeature`（`device.ts:160-174`）の `undefined` を「要求しない」に変える
（`shaderF16` と同じ規律 — `device.ts:370-` が既に「既定の意味が違う」と明記している）。
あるいは feature 取得は自動のままにして、**1 dispatch = 1 pass に開くかどうかを別の明示ノブ**にする。

- **期待利得（実測ベース）**: **−370〜375ms/step・解像度非依存**（1024px 375.2 / 512px 369.8）。
  さらに **step ごとの単調増加（step 2→10 で +235ms）が消える**。
- **工数**: **S**（`device.ts` の 1 分岐 + ADR 0021 の改訂 + デモ/ベンチ側の明示 opt-in）。
- **触るファイル**: `src/gpu/device.ts`、`docs/decisions/0021-gpu-timing-diagnostics.md`、
  `examples/anima/main.ts`・`examples/sbv2/main.ts`（計測したいときは `gpuTiming: true`）。
- **リスクと既存不変条件の判定**:
  - **ADR 0021 の設計自体は正しい**（pass 境界の timestamp が唯一の移植可能な計測点）。
    変えるのは**既定値だけ**で、`true` を渡せば従来どおり。
  - **診断の可用性**: 今まで無償で付いていた `lastRunTiming` が既定で `undefined` になる。
    **これは perf 作業の主要な観測手段**なので、ベンチ経路は明示 `true` に直す必要がある
    （デモの `.json` 出力が欠ける — `main.ts:743` は既に「有効なら」の形なので構造変更は不要）。
  - **測定の妥当性が上がる副次効果**: 現状は**計測装置が壁時計の 16% を占めており、
    「計測すると遅くなる」状態**。今後の perf 実測の基準としても既定 off の方が健全。
  - **ADR 0004 / errorScope / full-write / codegen 決定性**: **全て無関係**。

---

## 6. 推奨

### 推奨 = 案 1（+ 直後に案 3 を S 工数で同梱）

**根拠**:

1. **利得が桁違いに大きく、実機で確認済み**。1024px 非 GPU 705.6 → 161.9ms（2 回とも再現）、
   512px 壁 1,050 → 632ms/step。案 2（−139〜191ms）・案 3（−375ms）を足しても届かない。
2. **根治である**。案 2 と案 3 は「ホスト側の絶対量を減らす」対症で、**「ホスト代が GPU 時間に
   足し算で乗る」という構造そのもの**は残る。案 1 はその構造を壊すので、以後の dispatch 増加
   （案γ 波 2 の変換 dispatch、attention a8 の +224/step など）が壁時計に直撃しなくなる。
   正本 doc §3.4 ③ が「案γ は非 GPU 固定費を悪化させるリスクだけがある」と書いた制約も外れる。
3. **既存不変条件との衝突が最小**。errorScope 直列化・flush-before-destroy・full-write・
   codegen 決定性のいずれにも触れない。触れるのは ADR 0004 の適応制御の**観測点だけ**で、
   不変条件 1〜3 は保ったまま移植できる（しかも推定は安全側に倒れる）。
4. **他の 2 案の利得と重ならない**（案 3 は 512px 以下で残り、案 2 はホスト律速モデルで残る）
   ので、案 1 を先に入れても後続の判断材料が失われない。逆順にすると案 2/3 の効果が
   1024px で測定不能になる。
5. **移植性を落とさない**（ブラウザでは元から非ブロッキング）。

**縮退点**: 適応制御の再設計が重いと判断したら、**「N チャンクに 1 回だけ `onSubmittedWorkDone`
を呼ぶ」**（例: 窓の最初と最後）へ縮退できる。ブロック回数が 47 → 2 になれば利得の大半は取れる
（**試算**: 直列化される GPU 待ちが 47 分割 → 2 分割になるだけなので、ホスト代の重なり率は
1 − 2/47 ≈ 96%）。

### 次タスク（③ VAE tiling / DiT S 化）との優先度

**案 1 を先にやる価値がある**と考える。理由は 3 つ:

- 1024px turbo 全体 27.6s のうち **DiT の非 GPU だけで 7.1s（10 step × 706ms）= 25.7%**。
  VAE tiling が狙う VAE decode は 1.7s（6.2%）で、**桁が 1 つ違う**。
- 工数が M（1 ファイル・1 機構）で、**ADR も 1 本で済む**（ADR 0004 の適応制御の追補）。
  VAE tiling は graph 作業ゼロとはいえ資産の再生成と E2E 再導出が付く。
- **③ 完了後に予約されている最終 perf ベンチ**（1024×1024・8 step・方式×フェーズの VRAM/時間）
  の**測定基盤そのものが今は歪んでいる**（計測装置が壁の 16%・step で単調増加）。
  案 1 + 案 3 を先に通せば、そのベンチが素直な数字になる。

---

## 7. 開示・限界・未解決

1. **ハーネスの入力はダミー値**（形だけ実物と同じ）。dispatch 数・チャンク数・バッファ本数は
   実デモと完全一致（3,311 / 3,356 / 3,835）し、非 GPU も実デモ 693ms と一致したので
   帰属には足りるが、**GPU の絶対時間は denormal などで実データと微差がありうる**。
2. **`--stub-onswd` の擬似実験は適応制御を歪める**（§4.4 の開示）。案 1 の利得は
   「上限に近い実測」であって、実装後の値ではない。
3. **`encoder.finish` の内部機序は推測**（wgpu のトラッカー確保）。**実測で確定しているのは
   「pass 数に線形」「累積バッファ数に線形」「生存数ではない」の 3 点**まで。
4. **熱によるクロック変動**が全ての壁時計に乗る（同じ 1024px 構成で GPU pass 和が
   1,339〜1,762ms まで振れた run がある）。本 recon の主張は全て
   **同一 run 内の差分か、非 GPU 残差の比較**で立てており、run をまたいだ壁時計の直接比較は
   結論に使っていない。
5. **VAE decode 段・text 段の非 GPU は測っていない**（DiT のみ）。VAE は 395 dispatch/run なので
   同じ単価なら非 GPU は数十 ms 級（**試算**）。
6. **`buffer.mapAsync[resolve]` の扱い**: 47 本を同時発行するため素の合計（496.55ms）は
   多重計上。1 本ぶん（10.56ms）を実効値として計上した。ここだけ「合計」ではなく「窓」を
   採っている。
7. 未検証: 案 1 実装後に**キューが深くなることで VRAM やドライバ側にどういう影響が出るか**
   （command buffer の滞留）。TDR は per-submit なので理屈上は無害だが、実測はしていない。

---

## 8. 出典

**Karume（file:line）**: `src/runtime/executor.ts:731,974-982,985-1024,1010-1017,2333-2337,2340-` /
`src/gpu/submit.ts:297-313,342-349,417-462,441,471-512,522-541,568-574` /
`src/gpu/arena.ts:11-22,101-117,119-133,250-267` / `src/gpu/pipeline-cache.ts:43-52` /
`src/gpu/device.ts:155-174,352-370,558,658-687` / `examples/anima/main.ts:583,639-700,743`。

**ADR / doc**: [0004 実行モデル](/home/developer/workspace/karume/docs/decisions/0004-execution-model.md) /
[0021 GPU 時間診断](/home/developer/workspace/karume/docs/decisions/0021-gpu-timing-diagnostics.md) /
[0031 attention score f16 格納](/home/developer/workspace/karume/docs/decisions/0031-attention-score-f16-storage.md) /
`docs/research/2026-08-04-intermediate-f16-design.md` §2.3・§3.4 ③・§10-2。

**本日の実測ログ（scratchpad）**: `timedfull1.log` / `timedfull2.log` / `timedoff1024.log` /
`run_timed_on_512.log` / `run_timed_off_512.log` / `run_off_on_1024.log` / `run_off_off_1024.log` /
`timedA1.txt` / `timedA2.txt`、および `micro_onswd.ts` / `micro_finish.ts` の標準出力。
