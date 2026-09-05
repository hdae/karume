# 0005 — 検証戦略（Deno GPU CI を一級成果物に）

- Status: accepted（2026-08-01）
- 根拠資料: recon §1/§8/§9-2。先行実験プロジェクト（以下プロトタイプ）最大の検証の穴は
  「WGSL カーネルの数値検証が自動テストに無い」ことで、プロトタイプ自身が Deno CI を
  本筋の解と明記。本環境（Deno 2.9.4 + RTX 3080 Ti、
  フラグ無しで WebGPU 動作、shader-f16 / timestamp-query 有効）で実行可能なことを実測済み。

## 決定

検証は三段 + 受け入れの構成とし、**実 GPU の数値検証を CI（`deno test`）に含める**。

1. **WGSL スナップショット** — codegen 決定性（バイト単位同一）を固定する。
2. **CPU 参照実装との allclose** — 判定は常に `|x−y| ≤ atol + rtol·|ref|`
   （相対誤差のみは禁止）。NaN / Inf は不合格。GPU 不要でどこでも走る。
3. **実 GPU golden 突合**（Deno + 実ハードウェア）— op 単位・グラフ単位で torch 由来の
   ゴールデンと突合する。op を 1 個足すたびにここへテストを足すことを実装契約にする。
4. **実ブラウザ受け入れレーン** — examples/ 側のハーネスで手動実行（自動化しない）。
   Dawn ≠ wgpu の実装差（丸め・features）はここで受け止める。

### 規律（プロトタイプから継承）

- 全ケース SKIP は明示 FAIL（無音の見かけ成功を防ぐ）。
- エクスポータの emit 集合 ⊆ ランタイム実行可能集合を**dtype 組込みの契約テーブル**で
  突合する（op 名のみの突合は f32 sum 事故を見逃した）。
- 「型が通った」≠「同じ数値が出る」— 移植・カーネル変更は必ず段 3 を通す。
- GPU テストは環境により縮退する（アダプタ無し環境では段 3 を SKIP と明示表示するが、
  リリース判定は段 3 緑を必須とする）。

## 帰結

- CI マトリクスに「バックエンド × dtype 経路」の緑条件を定義する必要がある
  （lavapipe 等ソフトウェアアダプタは f16 を f32 計算するため、実 HW レーンと区別する）。
  具体値は M0 実測後に追記する。
- 追記（2026-09-05）: リリース判定機は `shader-f16` と `timestamp-query` を列挙するアダプタで
  あること。列挙しない機で verify を通すには `KARUME_ALLOW_NO_SHADER_F16=1` /
  `KARUME_ALLOW_NO_TIMESTAMP_QUERY=1` を意図表明として設定する（既定は
  `packages/runtime/tests/gpu_gate_test.ts` が全 SKIP を FAIL にする）。実資産不在も同形で
  `assets_gate_test.ts` が FAIL にし、opt-out は `KARUME_ALLOW_NO_ASSETS=1`。これらを設定した
  環境の緑はリリース判定に使わない（[release-runbook](../release-runbook.md) §1）。
