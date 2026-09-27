> 2026-09-27 時点の調査スナップショット。Intel Arc B570（BMG G21・VRAM 9.93 GiB・ReBAR）/ Linux xe / Mesa ANV 25.0.7 /
> Deno 2.9.6（wgpu-core 29.0.1・wgpu-hal 29.0.3・gpu-allocator 0.28.0）。ソースは crates.io の tarball（Deno v2.9.6 の
> Cargo.lock のチェックサムと一致）・deno v2.9.6 の `ext/webgpu`・Dawn main `6c1e2771`。台本・JSON・保存したソースは
> `outputs/bench/karume/2026-09-27_h35/`（git 追跡外）。

# H-35: OOM 退避のやり直しが device lost になる原因 — OOM には 2 種類ある

perf-ledger H-35（anima の DiT 常駐で、OOM を退避したあとのやり直しが B570 で `GpuDeviceLostError` になる）の原因調査。
結論は「**`createBuffer` の OOM だけが回復できる OOM で、`queue.writeBuffer` の staging で出る OOM は device を
道連れにする。しかも両者は JS から同じ `GPUOutOfMemoryError` に見え、消失は次の呼び出しまで通知されない**」。
OOM を踏んでから退避する設計（ADR [0112](../decisions/0112-anima-transformer-residency.md) 決定 3）は、踏んだ OOM が
staging 側だった時点で手遅れになる。対処は ADR 0112 の追記（測った空きで先に退避する）。

## 1. 実測（素の WebGPU・製品コード無関係）

台本 `oom-kind-probe.ts`。1 プロセス 1 device（device を破棄して作り直すと確保できる総量が減る — known-issues
「Intel Arc B570」節）。1 GiB の STORAGE バッファ（書き込まない）を OOM まで積む（毎回ちょうど 9 本）と、続く
アップロードの形（`createBuffer` → `writeBuffer`・64 / 256 MiB・8 item ごとに `submit([])` + `onSubmittedWorkDone`）で
最初に失敗する呼び出しを見た。各シナリオ 2〜3 走行・結果はすべて同一。

| シナリオ                                         | 最初に失敗した呼び出し    | errorScope             | device                                              | 全解放してやり直し |
| ------------------------------------------------ | ------------------------- | ---------------------- | --------------------------------------------------- | ------------------ |
| fill-submit（createBuffer が先に OOM）           | createBuffer              | out-of-memory          | 生存（submit・fence・mapAsync・再確保すべて通る）   | 成功               |
| fill-created / headroom-1g                       | writeBuffer（item 1 / 2） | out-of-memory          | 直後に無効。lost は次の submit で +0.3 ms           | 失敗               |
| write-oom-freeall                                | writeBuffer               | out-of-memory          | 約 9.1 GiB を全部 destroy した後の submit でも lost | 失敗               |
| staging-burst（256 MiB × 12 を同スコープで連打） | 9 本目の writeBuffer      | **何も出ない（null）** | 無効。lost は fence の submit で +0.3 ms            | 失敗               |
| staging-burst-split（1 本 1 スコープ）           | 9 本目の writeBuffer      | out-of-memory          | 同上                                                | 失敗               |
| write-oom-map                                    | writeBuffer               | out-of-memory          | mapAsync が 1 ms で reject し同時に lost 解決       | 失敗               |

- `writeBuffer` の OOM の後は、`device.lost` が **タイマー（50 ms / 1000 ms）でも submit なしの `onSubmittedWorkDone` でも
  解決しない**。検証を通る呼び出し（submit・writeBuffer・createBuffer・mapAsync）を 1 つ出すと 0.2〜0.3 ms で解決する
  （反証レッグの確認: submit 0 回のまま 4 byte の writeBuffer を出しても +0.2 ms で解決）。
- 同じスコープで writeBuffer を重ねると、2 本目以降が「device が無効」で落ち、Deno はその時点で `is_lost` を立てて
  以後の `popErrorScope` を null にする — **OOM そのものが報告されない**形になる。
- karume の 2026-09-26 の実走（evict-probe・ダミー 6 GiB）: `evicted` / `out-of-memory` の 2.6 ms 後に、やり直しの
  part 1（7 item・1.6 MB）のフェンスで `GpuDeviceLostError`。消失はやり直しより前に起きていた。どの呼び出しが OOM を
  出したかの記録は無い（最有力は staging・反証レッグの判定は uncertain）。ただしどちらの経路でも「通知なしに無効 →
  次の呼び出しで lost」の同じ形になり、設計上の結論は変わらない。

## 2. 機序（ソース）

### wgpu-core / wgpu-hal（Deno）

- `Device::handle_hal_error` は hal の `OutOfMemory` / `Lost` / `Unexpected` すべてで `self.lose()`（device/resource.rs:702-718）。
  `handle_hal_error_with_nonfatal_oom` は `OutOfMemory` だけを返す（lose しない）。**非致命の方を使うのは
  `create_buffer` / `create_texture` / `create_sampler` / `create_query_set`（と ray tracing）だけ**（resource.rs:1100, 1629,
  2244, 4924）。
- `StagingBuffer::new`（resource.rs:1120-1132）は致命の方を使う。呼び手は `queue.write_buffer`（queue.rs:545）・
  `write_texture`・`create_staging_buffer`・**MAP_WRITE を持たない `mappedAtCreation`**（resource.rs:1159）。
  失敗は `QueueWriteError::Queue(DeviceError::OutOfMemory)` = ErrorType::OutOfMemory として返る（lose した後）。
- submit・command encoder・bind group / pipeline 生成・map も致命の方（queue.rs:396-401, 1452 / command/mod.rs:664-779 /
  resource.rs:2496, 2824, 3437, 3914, 4675）。
- OOM の判定はドライバの確保失敗ではなく **wgpu の事前予算チェック**（wgpu-hal vulkan/device.rs:779-866 —
  `heap_usage + size >= budget × 97%`・Deno が `lib.rs:301-304` で 97 / 99 を焼き込む・wgpu PR #7472）。host access が
  要るバッファ（MAP_*）は host-visible な heap 全部、それ以外は device-local な heap を検査する。**submit と poll の
  たびに `lose_if_oom`（使用量 ≥ 99% で lose）**が走る（queue.rs:1503・resource.rs:801）。poll では maintain（解放）→
  判定の順、submit では submit → 判定の順（解放は次の poll）。
- gpu-allocator の `CpuToGpu`（MAP_WRITE の staging）は `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` を優先する
  （vulkan/mod.rs:803-809）。B570 では heap 0（VRAM 9.93 GiB）がその種類（type 3 / 6）を持つので、**staging は VRAM に
  置かれ、予算チェックも heap 0 を見る**（`vulkaninfo.txt:949-1052`）。
- Mesa ANV の budget は `MIN(heap_size, heap_used + 空き × 0.9)`（anv_physical_device.c:2848-2905）— 他プロセスの
  使用量でも動く。

### deno_webgpu

- `writeBuffer` の OOM は `push_error` で out-of-memory スコープへ（queue.rs:193-198・error.rs:77-147）。致命か非致命かの
  区別は JS に届かない。
- **wgpu の `device_lost_closure` を登録していない**（error.rs:201 の TODO・`ext/webgpu` に `set_device_lost` の呼び出し
  なし）。`device.lost` は後の呼び出しが `ErrorType::DeviceLost` を返したときだけ解決する（reason `unknown`・文言
  "device was lost"）。以後 `popErrorScope` は null（device.rs:740-748）。
- エラー経路で device を destroy / drop する箇所は無い。

### Dawn（Chrome）

- `Device::HandleError` は Validation とその呼び出しが許した型だけを errorScope へ通し、それ以外は device lost に
  変える（Device.cpp#L643-654）。OOM を許すのは createBuffer / createTexture / createQuerySet / createResourceTable。
  `writeBuffer` の staging（DynamicUploader・4 MiB 超は専用の mappedAtCreation バッファ）・submit・bind group 生成の OOM
  は device lost。規則は wgpu と同じ。
- ただし Dawn の memory type 選択は staging を **最大の heap（この機ではシステム RAM 31 GiB）**に置く
  （MemoryTypeSelector.cpp#L72-159）ので、VRAM が尽きたときの OOM は createBuffer 側で出やすい（推論・未実測）。
  D3D12 は residency 管理で createBuffer が超過を通し、submit 時の MakeResident 失敗で device lost。Metal は command
  buffer のエラーで device lost。
- WebGPU 仕様が保証するのは「createBuffer の確保が副作用なく失敗したら out-of-memory」と「他の資源を解放すれば
  通りうる」だけ（spec L3693-3695・L15744-15747）。writeBuffer に OOM の段は無い。

## 3. 棄却した修正案（実測）

**明示 staging**（`createBuffer(MAP_WRITE | COPY_SRC, mappedAtCreation)` → `getMappedRange().set` → `unmap` →
`copyBufferToBuffer`。確保が全部 createBuffer になるので OOM が非致命になる）: 台本 `upload-path-probe.ts`・
256 MiB × 4 = 1 GiB・ABAB × 3 の中央値。

| 経路                           | 1 GiB の壁時計 | 内訳                                                                   |
| ------------------------------ | -------------: | ---------------------------------------------------------------------- |
| A: `queue.writeBuffer`（現行） |        0.277 s | writeBuffer 発行 240 ms・フェンス 35 ms                                |
| B: mappedAtCreation + copy     |       82.583 s | **`getMappedRange` 82,095 ms**・set 54 ms・unmap 194 ms・create 217 ms |

Deno の `getMappedRange` は map した領域を **V8 側へ丸ごと複製**する（buffer.rs:269-300 `slice.to_vec()`）。ReBAR の
VRAM を CPU が読む速度（約 12 MB/s）がそのまま出るので、Deno では選べない（Chrome は zero-copy で事情が違う）。

## 4. 対処（ADR 0112 追記・2026-09-27）

- **測った空きで先に退避する**: runtime に `fitsHeadroom(gpu, bytes)`（`createBuffer` の非致命な OOM で「今この量が
  入るか」を 1 度だけ試す — `packages/runtime/src/gpu/headroom.ts`）を足した。STORAGE バッファを `maxBufferSize`
  以下の等分に割って確保し、即 destroy して `onSubmittedWorkDone`（device 消失と競わせる）で解放を確定させてから
  返す。中で `submit([])` は出さない（submit 後の 99% 線の判定は解放より先に走る）。
- anima は常駐 DiT があるとき、`TransformerResidency.ensureHeadroom(need, probe, notify)` を generate の中の 2 点で
  呼ぶ:
  - text 段の前（持ち越した常駐 DiT があるとき）: need = text_encoder / text_conditioner の必要量の大きいほう。
  - `transformer` の `stage` end の後・`vae_decoder` の start の前（その時点で常駐 DiT があるとき）: need =
    vae_decoder の必要量。最初の generate（その generate で作った DiT の上に VAE が乗る）と、持ち越した DiT が
    新しい解像度で backing を育てた generate を覆う。
  - 段 1 本の need = `ModelComponent.estimate(...).peakAccountedBytes` + `ModelComponent.maxPartBytes`（最大の
    重み part = その part のフェンスまで残る staging）+ `HEADROOM_MARGIN_BYTES`（512 MiB）。
  - 入らなければ、格下げ（OOM の退避と同じく sticky）→ 常駐 DiT を破棄 → 解放待ち → イベント `evicted` /
    `headroom`。試し確保が投げたら（validation / device 消失）格下げせずにそのまま投げる。反応の退避（OOM を
    踏んでから）は第二線として残す。
- **無効化された device を早く表面化させる**: 退避の解放待ち（`settleReleasedMemory`）で待つ前に `queue.submit([])` を
  1 回出す（検証を通る呼び出し = Deno が lost を解決する唯一の契機。pending destroy も flush される）。
- 余裕 512 MiB の根拠: gpu-allocator のブロック 256 MiB（ブロック未満の要求は事前チェックが見た量より大きい実確保に
  なりうる）+ 99% 線までの幅（予算の 2% — この機で約 178 MiB）+ estimator の未計上分。

## 5. 併せて判明した古い記述

- 「VRAM 自体が返るのは `device.destroy()` のみ」（arena.ts / device.ts / context.ts の 3 か所）は B570 の probe と
  食い違う: `buffer.destroy()` + `onSubmittedWorkDone`（= poll）で次の確保に返る（2026-09-26 の raw probe・本調査の
  fill-submit / 試し確保のテスト）。3 か所とも `ca9af567` で直した。
- ADR 0070 決定 5 は staging を estimator が数える項目に挙げるが、実装は `unaccounted` に置いている（estimate.ts:291-297）。
  ADR 0070 の 2026-09-27 追記に既知の食い違いとして記録した（先回りの必要量は最大 part を明示的に足す）。

## 6. 修正後の実測（B570）

- 再現台本 `outputs/bench/karume/2026-09-26_anima-residency/evict-probe.ts --dummy-gib 6`（常駐 DiT の上で共有
  device に 6 GiB のダミーを積んで 2 枚目を生成する）: 1 枚目 23.44 s（`retained` / `request`）。2 枚目 23.46 s —
  generate 開始から 0.09 s（`text_encoder` の `stage` start の前・トークン化を含む）で `evicted` / `headroom`、DiT 段で
  `released` / `downgraded`、生成は成功。PNG の sha は 1 枚目と一致（`16f7946acc33`）。OOM も device lost も出ない。
- e2e `--filter residency`: 4 行緑。新しい 2 行 = text_encoder の前の退避（空きを 1 GiB 程度残す形）と、最初の
  generate の VAE 段の前の退避（空き 256 MiB 未満）。
- 試し確保の費用: 512 MiB の `fitsHeadroom` 単体で 31 ms。常駐 DiT がある generate は 2 回量る（0.1 s 程度）。
- **M2（利用者・Apple M2 24 GB・Chrome 153・metal-3・確認ページ `tools/anima-residency/browser`・512²・seed 42・
  `outputs/bench-browser/anima-residency-browser-2026-09-27T12-19-09.299Z.json`）**: 常駐 on の 4 generate は
  1 枚目 234.4 s（text_encoder 43.6 / conditioner 12.0 / DiT 168.3〈初回ロード込み〉/ VAE 10.0 s）、2〜4 枚目 146.5〜
  147.6 s（DiT 79.6〜80.0 s）で、毎回 `retained` / `request`・PNG sha は 4 枚とも一致（常駐の利得 ≈ 87 s /
  generate）。ダミーは 1 GiB × **80 本（85,899,345,920 B）まで 1 度も OOM にならず**、その上での 4 枚目も退避なしで
  通った（146.5 s・sha 一致）。Metal（Dawn）は `createBuffer` で物理メモリを裏付けないため、試し確保もダミーも
  「入る」としか答えず、**この機では先回りの退避は起きない**（known-issues「Metal で out-of-memory errorScope が
  沈黙する」節と同じ性質 — 書き込んで初めて圧が掛かる）。連続生成の利得と決定性は設計どおり。
- B3 ベンチ（ABBA・n=8・1024² turbo・`bench.ts --count 4`・2026-09-27）での試し確保 2 回込みの壁: 常駐あり 中央値 **20.86 s**（前日の試し確保なし 21.05 s）・段ごと運転 23.52 s（前日 23.49 s）→ 利得 2.66 s / 11.3%（前日 2.45 s / 10.4%）。試し確保の費用は走行間のばらつきに埋もれる（PNG sha は 20 走行とも一致・`results-2026-09-27T10-54-12Z.json`）。
