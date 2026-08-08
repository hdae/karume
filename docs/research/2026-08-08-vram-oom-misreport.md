# VRAM 不足が「破棄後使用」に化ける誤報告 — 根因と修正 recon

> NOTE: 時点スナップショット。文中のパス・行番号・実測値は記録当時（2026-08-08）のリポジトリ
> 構成と実機（RTX 3080 Ti 12,288MiB / Deno 2.9.4 / wgpu 29.0.1 / x86_64-linux）に基づく。

発端は「Anima の PNG 門 4 本のうち **f16-1024 だけが間欠で落ちる**」という観測。文言は
`GpuValidationError: run のエンコード: Buffer with '' label is invalid` で、素直に読むと
「破棄済み / 無効なバッファを bind group に入れた」= ライフサイクルのバグに見える。実際は
**VRAM 確保の失敗（OOM）が派生 validation に化けていた**もので、破棄後使用ではなかった。

## 1. 機序

1. VRAM の余力が切れた状態で `createBuffer` が呼ばれる。
2. WebGPU では確保失敗は**同期例外にならない** — out-of-memory スコープにエラーが入り、
   呼び出し側には**無効なバッファ**が返る。
3. その無効バッファを `createBindGroup` に渡すと、今度は **validation** スコープに
   `Buffer with '' label is invalid` が入る（根因の派生）。
4. `popFailureScopes` は validation を先に見て返していたため、根因の `GpuOutOfMemoryError` が
   捨てられ、破棄後使用と区別のつかない文言だけが残る。

errorScope は 1 スコープにつき最初の 1 件しか保持しないため、上流でどれだけ派生が起きても
弁別材料は増えない。

## 2. wgpu のメッセージは 2 症状を弁別する

同一 device 上で 3 通りの状態を作って捕捉した文言（probe による実測）:

| 注入した状態                                                | validation スコープ                       | out-of-memory スコープ   |
| ----------------------------------------------------------- | ----------------------------------------- | ------------------------ |
| destroy 済み → `createBindGroup` / `submit` / `writeBuffer` | `Buffer with '' label has been destroyed` | null                     |
| 巨大 `createBuffer`（確保失敗）                             | null                                      | `not enough memory left` |
| 失敗して返った無効バッファ → `createBindGroup`              | `Buffer with '' label is invalid`         | null                     |

観測ログの `is invalid` は 3 行目でしか出ない。**`has been destroyed` と `is invalid` は別物**で、
前者だけが本当のライフサイクル違反。

## 3. 確定再現（フォールト注入）

`tools/diag/hold-vram.ts` で別プロセスに **4,608MiB** を保持したまま同じ 4 連を回す:

```
deno run -A tools/diag/hold-vram.ts 4608          # 端末 A（Ctrl-C まで保持）
deno test -A packages/models/tests/e2e_anima_test.ts   # 端末 B
```

結果は w8a8/1024 ok (20s) / w8a8/512 ok (8s) / **f16/1024 FAILED (8s)** — 観測ログと同一の文言・
同一のスタック（`popFailureScopes` → `#runOnce`）/ fromPretrained ok (13s)。無負荷の同一 4 連は
全緑（f16-1024 は 40.4s で通過）。**保持量を振れば「ピーク + 保持量 > 天井」の線が測れる**ので、
修正の効果測定にもそのまま使える。なおこの再現は **F2 適用前**のもの — 適用後は初回ピークが
2.6GiB 下がるので、同じ 4,608MiB では落ちなくなる見込み（保持量を上げれば同じ形に持ち込める。
再実測は未実施）。

## 4. なぜ f16-1024 だけか — メモリプロファイル

nvidia-smi 1Hz（無負荷の 4 連・**修正前**）:

| ケース          | ピーク                            |
| --------------- | --------------------------------- |
| w8a8-s16 / 1024 | 3,421MiB                          |
| w8a8-s16 / 512  | 2,818MiB                          |
| **f16 / 1024**  | **瞬間 8,391MiB → 定常 5,723MiB** |

テスト間は 338/55MiB まで回収される（跨ぎのリークではない）。同機の `createBuffer` 天井は
**11,136MiB**（`hold-vram.ts` で 256MiB 刻みに確保 → `not enough memory left`。同日の別サンプル
では 11,264MiB — `heapBudget` が動的なので幅が出る）なので、余裕は 2.7GiB — 他プロセスが
2.7GiB 以上使えば落ちる。f16 preset の transformer は 3,913,665,620B で i8 系列の約 2 倍、
活性は両者とも f32 なので差は重みだけで説明が付く。

瞬間 8,391 → 定常 5,723MiB の落差 2,668MiB の正体は **重み staging の二重計上**だった。
`Session.create` は全 initializer を `queue.writeBuffer` で上げるが、**submit を 1 度も挟まずに**
Session を返していた（`executor.ts` の push → writeBuffer ループ → pop → return）。probe の実測:

- `createBuffer` 2GiB 後 2,311MiB → `writeBuffer` 後 **4,359MiB**（staging が別勘定で乗る）
- `onSubmittedWorkDone` だけ（submit 無し）→ 4,359MiB のまま解放されない
- **`queue.submit([])` + 完了待ちで 2,306MiB へ解放**

つまり重み 3.7GiB ぶんの staging が「最初の run の submit が完了するまで」VRAM に残っていた。

**落とし穴: `SubmitScheduler.flush()` では submit が出ない** — pending dispatch が空だと
`#submitChunk()` が即 return するため、staging は溜まったまま。吐かせるには実際の
`queue.submit([])` が要る。

## 5. 修正

### F1 — `popFailureScopes` の優先度反転（`gpu/device.ts`）

両スコープが捕捉されたときは **out-of-memory を先に返す**。validation 側は常に派生
（無効バッファの使用）で、根因は必ず OOM 側だから。純 validation のみのときの挙動は不変。
フェイク device で両スコープ同時捕捉を注入する単体テストで固定した。

### F2 — 重みアップロード後の実 submit（`runtime/executor.ts`）

`popFailureScopes` の直後・Session を返す前に `queue.submit([])` + 完了待ち
（`raceDeviceLost` 経由 — device 消失時に `onSubmittedWorkDone` は解決しないため）を 1 回入れ、
staging を吐かせる。コストは Session 生成ごと 1 回の完了待ちだけ（run のホットパスではないので、
「submit ごとに `onSubmittedWorkDone` を呼ばない」という submit.ts の計測の帰属にも触れない）。
この submit は errorScope で囲んでいない — 空の submit は確保も検証も伴わず、かつ
`Session.create` は `GpuContext` のスコープロック外なので、await を跨ぐスコープをここに張ると
並行 Session の失敗を誤帰属させる口になるため。

**効果（実測）**: f16-1024 の PNG 門を単独実行（nvidia-smi 1Hz）してピーク **5,723MiB** —
修正前の 8,391MiB から **2,668MiB 減**。トレース上、初回の瞬間スパイクは消えて定常値がそのまま
ピークになった。天井 11,136MiB に対する余裕は 2.7GiB → **5.4GiB** に広がる。

### F3 — 再現の常設（`tools/diag/hold-vram.ts`）

引数の MiB 数（既定 4,608）を 256MiB 単位の `GPUBuffer` で確保して Ctrl-C まで保持するだけの
スクリプト。確保に失敗したら「どこまで確保できたか」を添えて fail loudly させているので、
天井そのものの実測にも使える。

## 6. 弁別のしかた（今後の読み方）

- `has been destroyed` → 本当にライフサイクルのバグ。
- `is invalid` → **上流の確保失敗を疑う**。F1 以降は `GpuOutOfMemoryError` が先に立つので、
  それでも `is invalid` が出るなら確保以外の無効化を探す。
- 症状が「間欠の device 消失」に化けている場合は 99% 線側
  （[2026-08-03-wgpu-memory-ceiling.md](2026-08-03-wgpu-memory-ceiling.md) §2-3）。天井付近では
  OOM ではなく device lost になる境界がある。

## 7. 隣接（今回のスコープ外）

- `RunArena.allocStorage` は確保直後の妥当性を見ずにバッファを配る。失敗は run 終端の pop まで
  表面化しないので、確保点で弾ければ原因の位置がもっと近くなる。
- `limitations.md` の「f32 DiT（重み 7,465MiB）は載らない」根拠は、天井が時点値である以上
  再検討の余地がある（未検証）。
- `Session.dispose()` / `Session.create()` は `GpuContext` のスコープロック外で GPU 操作を出す。
  逐次利用では露見しないが、並行 Session では誤帰属の口になりうる。
