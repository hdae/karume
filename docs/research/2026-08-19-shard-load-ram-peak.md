# shard 逐次ロードの RAM ピーク実測（ADR 0070 受入③）

> **性格**: 時点スナップショット（2026-08-19・RTX 3080 Ti 12,288MiB・Deno 2.9.4 /
> Linux）。ADR [0070](../decisions/0070-shard-loading-admission.md) 決定 3 の実装
> （`createSessionFromShards`）が「全量 ArrayBuffer 保持の RAM 二重持ち」を実際に解消して
> いるかの受入実測。数値は本機のドライバ・GC 挙動に依存する。

## 方法

- 資産: `outputs/series/minicpm5-1b-decode/model.safetensors`（f32・4,324 MB・232 テンソル）。
- 分割: 512MB 目標でサイズ順詰め → **8 shard**（最大 802MB — 最大単一テンソル
  〈embedding / lm_head 級〉より小さくはできない。グラフ shard = `karume_ir` + 先頭テンソル群）。
- 計測: 1 構成 = 1 プロセス。`/proc/self/status` の **VmHWM**（ピーク RSS 高水位標）を
  プロセス終端で読む（サンプリング不要・同期区間も漏れない）。Session 構築 + dispose のみで
  run はしない。分割・計測の台本は本文末尾。

## 結果

| 経路                                                 |    peak VmHWM | ロード時間 |
| ---------------------------------------------------- | ------------: | ---------: |
| 全量面 `createSession`（単一 4.3GB）                 | **8,368 MiB** |       7.2s |
| shard 面 `createSessionFromShards`（8 shard）        | **3,463 MiB** |       6.9s |
| shard 面 + shard 境界で明示 GC（`--expose-gc` 診断） | **2,692 MiB** |       6.7s |

## 所見

1. **全量面のピーク ≈ 2 × ファイルサイズ**（8.4GB ≈ 4.3GB × 2）。内訳はホスト側の全量
   ArrayBuffer + `queue.writeBuffer` が submit 完了まで抱える実装 staging — ADR 0070 の
   Context が名指しした「配布ファイル全量 + GPU 常駐の RAM 二重持ち」そのもの。
2. **shard 面は −59%（−4.9GB）**でロード時間は落ちない（フェンス毎 shard の直列化コストは
   ディスク読みに隠れる）。ファイル総量への比例が切れ、ピークは「最大 shard + 定数倍」へ。
3. 定数倍の内訳: 明示 GC で −771 MiB（＝手放した shard バッファの回収遅れ）。残り
   ~2.7GB は baseline + 最大 shard（802MB）+ wgpu の staging 保持と V8 ヒープ拡張。
   **O(最大 shard) は漸近としては成立するが、係数は 3〜4 程度**（GC / ドライバ依存 —
   ランタイム側から縛れない）。
4. 下限は**最大単一テンソル**で決まる（802MB の表は shard に割れない — テンソル内分割は
   ADR 0070 のスコープ外）。
5. 受入③の判定: **合格**（全量比で「ファイル総量に比例しない」への転換を実測で確認）。
6. 追試（グラフ shard 参照の明示解放 = Codex RT-G-02 の修正後）: 3,464 / 明示 GC 2,690 MiB
   — **ピーク不変**。この機ではピークの内訳が「現在 shard + writeBuffer staging + 定数」で、
   グラフ shard のフレーム保持は実測に現れていなかった（V8 が未参照スロットを解放していた
   可能性）。修正は「エンジンのヒューリスティクス頼み」を「構造による保証」に置き換える
   もので、実測値の根拠ではない。

## 台本（再現用）

分割（`split_shards.ts` — 512MB 目標・整列降順詰め）:

```ts
// deno run -A split_shards.ts <入力.safetensors> <出力ディレクトリ> [目標shardバイト数]
import { parseSafetensors, tensorBytes } from ".../packages/runtime/src/format/safetensors.ts";
// 各 shard: ヘッダ JSON + データ節を整列降順（F32→I32→I4→F16→I8）で書き出し。
// karume_ir は先頭 shard の __metadata__ だけに載せる。f32 資産なので co-shard は自明。
```

計測（`measure_load.ts` — 1 プロセス 1 計測）:

```ts
// deno run -A measure_load.ts <whole|shards> <path>
const vmHwmKib = async () =>
  Number(
    (await Deno.readTextFile("/proc/self/status"))
      .split("\n").find((row) => row.startsWith("VmHWM:"))!.replace(/[^0-9]/g, ""),
  );
// whole: Deno.readFile → openModel → createSession
// shards: ディレクトリ内 shard-*.safetensors を辞書順に 1 本ずつ読む async generator →
//         createSessionFromShards（明示 GC 変種は yield 後に globalThis.gc?.()）
// 終端で VmHWM を出力。
```
