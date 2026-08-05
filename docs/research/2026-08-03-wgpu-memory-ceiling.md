# f32 天井（7,280MiB / 59.2%）の出所 — ソース特定 recon

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。

> NOTE: §7-2（limitations への移設）は同日実施済み — 現行の扱いは
> [../limitations.md](../limitations.md) の「Deno では GPUBuffer の総確保が…」節が正本。
> §7-1（vulkaninfo での budget 実測）は nix 供給を試行したが不発（2026-08-03: nixpkgs の
> vulkan-tools はローダがホスト NVIDIA ICD を読めず ERROR_INCOMPATIBLE_DRIVER — 非 NixOS +
> nix の典型問題で、nixGL 級の仕掛けかホストパッケージが要る）。ユーザー裁定により後回し。
> §7-3（上流提案）も未実施のまま。

- 日付: 2026-08-03 / 対象環境: Deno 2.9.4 (x86_64-unknown-linux-gnu), RTX 3080 Ti 12,288MiB
- 実測の前提: `docs/known-issues.md` 第1節（2026-08-02 の M1-P4 波3 実測）。再計測はしていない。
- 記法: **[一次]** = ソース/公式ドキュメントで確認、**[推論]** = 実測値と一次ソースからの導出、
  **[仮説]** = 未検証。

---

## 0. 結論（先に）

天井は Karume 側でも Deno の JS API 側でもなく、**Deno がハードコードしている wgpu の
「メモリ予算しきい値」** が入口。ただし **59.2% という数値自体は wgpu の定数ではない**。

```
確保可能上限 ≒ 0.97 × heapBudget(device-local heap)      ← createBuffer が OOM を返す線
device 消失  ≒ 0.99 × heapBudget(device-local heap)      ← submit / poll のたびに判定
```

`0.97 / 0.99` は Deno がハードコードした定数 **[一次]**。`heapBudget` は
**NVIDIA ドライバが `VK_EXT_memory_budget` で返す動的な値**であり、wgpu にも Deno にも
「60%」に相当する係数は存在しない **[一次]**。よって残る未解明点は「なぜこの機械の
heapBudget が約 7.5GiB（総量の 61%）なのか」に縮まる **[推論]**。

---

## 1. バージョンの確定 **[一次]**

- ローカル: `deno 2.9.4 (stable, release, x86_64-unknown-linux-gnu)`（`deno --version`）
- Deno のワークスペース Cargo.toml（v2.9.4 タグ）:
  `wgpu-core = "=29.0.1"` / `wgpu-types = "=29.0.1"`（`=` 固定）
  https://github.com/denoland/deno/blob/v2.9.4/Cargo.toml#L457-L458
- `ext/webgpu/Cargo.toml`: `wgpu-core` を `["trace","replay","serde","strict_asserts","wgsl","gles"]`
  ＋ unix で `vulkan` フィーチャ付きで依存。`wgpu-hal` は間接依存。
  https://github.com/denoland/deno/blob/v2.9.4/ext/webgpu/Cargo.toml
- wgpu v29.0.1 のリリース日: **2026-03-26**（GitHub Releases API）
- 現在の deno `main`（2026-08 時点）も同じ `=29.0.1` 固定・同じしきい値なので、
  Deno を上げるだけでは変わらない **[一次]**。

## 2. 出所のコード **[一次]**

### 2-1. Deno がしきい値を注入している箇所（ハードコード）

`ext/webgpu/lib.rs` の `get_or_init_instance`:

```rust
wgpu_types::InstanceDescriptor {
  backends,
  flags: wgpu_types::InstanceFlags::from_build_config(),
  memory_budget_thresholds: wgpu_types::MemoryBudgetThresholds {
    for_resource_creation: Some(97),
    for_device_loss: Some(99),
  },
  ...
```

https://github.com/denoland/deno/blob/v2.9.4/ext/webgpu/lib.rs#L298-L304
（ローカルで raw ファイルを取得し `grep -n` で行番号確認済み: 301-303 行が 97/99）

wgpu 側の型定義（`Option<u8>`、既定は `Default` 由来の `None` = しきい値なし）:
https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-types/src/instance.rs#L325-L337
→ **wgpu の既定では天井は無い。Deno が明示的に 97/99 を入れているのが起点。**

### 2-2. createBuffer 側（= 7,280MiB の OOM）

`wgpu-hal/src/vulkan/device.rs::error_if_would_oom_on_resource_allocation`
（v29.0.1: 779-868 行）https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-hal/src/vulkan/device.rs#L779-L868

要点:

1. `for_resource_creation` が `None` なら即 `Ok`（784-790 行）
2. **`VK_EXT_memory_budget` が有効でなければ即 `Ok`**（792-797 行）← 天井が消える唯一の分岐
3. `vkGetPhysicalDeviceMemoryProperties2` + `VkPhysicalDeviceMemoryBudgetPropertiesEXT` を
   **毎回**取得（799-816 行）
4. `needs_host_access`（MAP_READ/MAP_WRITE 有無）で host-visible heap 群か device-local heap 群かを選ぶ
   （818-848 行）。Karume の重み/活性は STORAGE のみ = `GpuOnly` → **device-local heap 群**
   （呼び出し側 895-903 行）
5. 判定式（**862 行**）:

```rust
if heap_usage + size >= heap_budget / 100 * threshold as u64 {
    return Err(crate::DeviceError::OutOfMemory);
}
```

https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-hal/src/vulkan/device.rs#L854-L865

呼び出しは `create_buffer`（904 行）、`create_texture`（1059 行）、`create_query_set`（2175 行）、
acceleration structure（2510 行）。つまり **バッファの作り方（1本のサイズ・アップロード経路・
flush 頻度）を変えても判定式に入るのは `heap_usage + requirements.size` だけ**で、
known-issues の「6 通り全部同値」という実測と完全に整合する **[推論]**。

### 2-3. device 消失側（= f32 DiT 7,465MiB で lost）

`wgpu-hal/src/vulkan/device.rs::check_if_oom`（v29.0.1: 2678-2726 行）

```rust
if heap_usage >= heap_budget / 100 * threshold as u64 {   // threshold = 99
    return Err(crate::DeviceError::OutOfMemory);
}
```

https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-hal/src/vulkan/device.rs#L2678-L2726
（こちらは heap の種別で絞らず **全 heap** を走査する点が 2-2 と違う）

呼び出し元 `Device::lose_if_oom`（wgpu-core）:
https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-core/src/device/resource.rs#L688-L699

```rust
/// Checks that we are operating within the memory budget reported by the native APIs.
/// If we are not, the device gets invalidated.
```

`handle_hal_error` が `OutOfMemory` を受けると **`self.lose(...)`（= device lost）** に変換する
（同 701-710 行）。`lose_if_oom` は **`Queue::submit` の末尾**（wgpu-core/src/device/queue.rs:1503）と
**`poll_and_return_closures`**（同 resource.rs:801）から呼ばれる。

→ known-issues の「OOM 例外ではなく **device 消失**、しかも `createSession` の途中で起きて
以後のテストを道連れ」は、この経路そのもの **[推論、整合性は高い]**。

## 3. 59.2% はどこから来るか — 数値の分解 **[推論]**

wgpu / Deno のどこにも 60% 相当の係数は無い **[一次: 全ソースを grep 済み]**。
実測から逆算すると:

| 観測                              | 判定式                                    | 逆算される `heapBudget` |
| --------------------------------- | ----------------------------------------- | ----------------------- |
| 7,280MiB は成功、7,296MiB で OOM  | `usage + size ≥ 0.97·B`                   | B ≈ 7,505〜7,522MiB     |
| 7,465MiB を積む途中で device lost | `usage ≥ 0.99·B` → 7,430〜7,446MiB で発火 | 同上と無矛盾            |

**強い傍証**: 2 つの観測の比 7,465 / 7,296 = 1.023 が、しきい値の比 99/97 = 1.0206 とほぼ一致する。
**単一の `heapBudget ≈ 7.5GiB` で OOM 線と lost 線の両方が同時に説明できる**ため、
「しきい値機構が出所」はほぼ確実 **[推論だが高信頼]**。

- B ≈ 7,505〜7,522MiB は総量 12,288MiB の **61.1〜61.2%**。
  59.2% は「0.97 × 61.1%」の結果であって、単独の定数ではない。
- **未解明**: なぜ NVIDIA ドライバがこの機械で 12GiB のうち 7.5GiB しか budget と報告するのか。
  現在 `nvidia-smi` は 33MiB/12,288MiB 使用（=ほぼ空）で、他プロセスの占有では説明できない
  **[一次: nvidia-smi 実行]**。Vulkan 仕様上 `heapBudget ≤ heapSize` であり、ドライバが
  OS 状況を見て自由に縮められる値なので、ドライバ実装依存の可能性が高い **[仮説]**。
  - 対抗仮説（未検証）: `heapUsage` の方が我々のバイト数より約 1.6 倍に膨らんでいる
    （budget≈heapSize のまま）。ただしバッファ 1 本 256MiB / 16MiB の両方で天井が同じだった
    実測から、アロケータのブロック padding では 1.6 倍は説明しづらい **[推論]**。
  - **決着方法（軽い）**: `vulkaninfo` は `VkPhysicalDeviceMemoryBudgetPropertiesEXT` の
    heapBudget / heapUsage を印字する。当機には未インストール（`command -v vulkaninfo` が空）。
    `vulkan-tools` を入れて 1 コマンド叩けば B の実値と heap 構成（device-local heap が
    いくつあるか）が確定する。これが最短の次の一手。

## 4. 既知報告・上流の動き **[一次]**

- **denoland/deno#35195**（open, 2026-06-13）
  「False-positive `GPUOutOfMemoryError` on discrete GPUs with small Resizable BAR heaps
  (wgpu v28 OOM bug)」— 報告者は AMD/RADV で 64MiB の確保が失敗。原因を
  「Deno が `for_resource_creation = 97` を設定 + wgpu が **全 device-local heap** を走査するため、
  256MiB の ReBAR heap に引っかかる」と特定している。
  https://github.com/denoland/deno/issues/35195
  → **我々のケースとは別症状**（当機は 7.2GiB まで通っているので、小さい ReBAR heap に
  引っかかってはいない）だが、**「Deno の 97% しきい値が実アプリで天井になる」既知報告**として
  そのまま流用できる **[推論]**。
- **gfx-rs/wgpu#9643**（merged **2026-06-10**, trunk）
  `error_if_would_oom_on_resource_allocation` を gpu-allocator のヒープ選択ロジックに寄せ、
  **実際に確保されるヒープ 1 つだけ**を見るよう修正。Fixes #8479, #9206。
  https://github.com/gfx-rs/wgpu/pull/9643
  - **v29.0.1 は 2026-03-26 リリースなので、この修正は入っていない**（当機の v29.0.1 ソースは
    旧ロジックのまま = 全 device-local heap 走査）**[一次: tarball を展開して確認]**。
  - ただし trunk の新ロジックも `GpuOnly` では最初に見つかった DEVICE_LOCAL メモリタイプの
    heap（= NVIDIA なら VRAM heap 0）を見るので、**device-local heap が 1 つしかない当機では
    天井は変わらない見込み** **[推論]**。
    https://github.com/gfx-rs/wgpu/blob/trunk/wgpu-hal/src/vulkan/device.rs#L894-L960
- **gfx-rs/wgpu#9745**（open, 2026-06-25）「Vulkan OOM model doesn't fall back to non-device-local
  heaps for `CpuToGpu` memory」— PR 9643 の作者自身が「不正確な heapUsage/heapBudget や、
  確保で実際に消費される量の見積り違いで、通るはずの確保を弾く可能性がある」と明記。
  関連として Mozilla bug 2040218（Firefox も同種のしきい値を使う）を挙げている。
  https://github.com/gfx-rs/wgpu/issues/9745
  → **「予算判定は誤検知しうる」ことは上流でも認識済み** **[一次]**。

## 5. 設定ノブの有無 **[一次]**

| 層              | ノブ                                                                                                                 | 当プロジェクトから使えるか                                                    |
| --------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| wgpu API        | `InstanceDescriptor.memory_budget_thresholds`（`None` にすれば判定ごと無効）                                         | **不可** — Deno が 97/99 をハードコード（lib.rs:301-303）。deno `main` も同じ |
| Deno 環境変数   | `DENO_WEBGPU_BACKEND`（`Backends::from_comma_list`）と DX12 コンパイラ用の env のみ。**しきい値の env は存在しない** | バックエンド切替のみ可                                                        |
| Deno CLI フラグ | `--unstable-webgpu` 以外に該当なし                                                                                   | なし                                                                          |
| Vulkan 側       | 判定は `VK_EXT_memory_budget` が**有効でなければ丸ごとスキップ**（device.rs:792-797 / 2687-2693）                    | 拡張を隠せれば天井は消える **[仮説]**                                         |

- `DENO_WEBGPU_BACKEND=gl` にすると GLES バックエンドになり、`MemoryBudgetThresholds` は
  D3D12 と Vulkan のみ対応（`wgpu-hal/src/gles/device.rs:1669` の `check_if_oom` は no-op）なので
  天井は消える **[一次]**。ただし WebGPU の compute を GLES で回すのは機能・性能面で大幅な後退で、
  Karume の用途では現実的でない **[推論]**。
- 拡張を隠す案（Khronos Profiles レイヤ等で `VK_EXT_memory_budget` を device extension 一覧から
  落とす）は**未検証の仮説**。当機にレイヤが入っているかも未確認。
- 確実なノブは **Deno をパッチビルド**（lib.rs の 97/99 を上げる or `None` にする）。
  上流に「env で上書き可能にする」提案を出す価値はある（#35195 が既に土台になっている）**[提案]**。

## 6. VRAM 総量への比例性 **[推論]**

- コード上、上限は `0.97 × heapBudget` であり、**VRAM 総量ではなく「ドライバ申告の budget」に比例する**
  **[一次]**。budget が総量の何割になるかは wgpu の管轄外。
- 当機の比率（budget/heapSize ≈ 61%）がドライバ共通の性質なら、24GiB 機では
  `0.97 × 0.61 × 24,576MiB ≈ 14.5GiB` → f32 DiT の 7,465MiB ＋活性は**十分載る** **[推論]**。
- 逆に「budget ≒ 空き VRAM」型の素直な実装なら 24GiB 機では約 23GiB まで通り、やはり載る **[推論]**。
- どちらのモデルでも 24GiB 機で f32 が載る予測は立つが、**61% が定数だという保証はない**ので
  「比例する」と断定はできない **[仮説]**。12GiB 機での budget 実値（§3 の vulkaninfo）を取り、
  さらに別容量の機械で 1 点取れば確定する。

## 7. 次の一手（提案のみ・本タスクでは未実施）

1. `vulkan-tools` を入れて `vulkaninfo` の heapBudget/heapUsage と heap 構成を 1 回だけ採る
   （§3 の分岐を潰す最短手。数分・非破壊）。
2. 1 で budget ≈ 7.5GiB が裏取りできたら、`docs/known-issues.md` の当該節を
   「Karume 側では回避不能な外部制約」として `docs/limitations.md` へ移し、
   出所（Deno lib.rs:301-303 + wgpu vulkan/device.rs:862）を明記する。
3. 上流提案: Deno に `memory_budget_thresholds` の env 上書きを入れる issue（#35195 に相乗り可）。
4. f32 系列の回帰は 24GiB 機か、しきい値を外したパッチビルドでのみ回す方針にする。
