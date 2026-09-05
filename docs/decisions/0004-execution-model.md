# 0004 — 実行モデル（静的形状・submit 分割・資源管理）

- Status: accepted（2026-08-01）
- 改訂（骨格は現行・以下の ①〜③ が後続 ADR で置換済み・④ は置換予告。①② は本文の該当箇所にも
  追記がある）:
  - ① **full-write** — ADR [0014](0014-layout-ops-full-write.md): 「全書きしない op は新品を
    要求」からカーネル側の全域書きへ反転（バッファ管理 ⑤）。
  - ② **submit スケジューラ** — ADR [0032](0032-deserialize-submit-and-optin-timing.md):
    per-submit `onSubmittedWorkDone` を廃止し、適応制御の観測窓を flush 単位へ再定義。
  - ③ **fence とバッファ寿命** — ADR [0054](0054-resident-loop-and-fence.md): ResidentTensor /
    BatchScope を第 4 の寿命クラスとして追加し、通常 run の fence を `mapAsync` 1 本へ集約。
  - ④ **置換済み（2026-09-05・ADR [0093](0093-transient-liveness-packing.md)）**: バッファ管理の
    **サイズ別プール**は区間 + offset 配置（生存区間の first-fit パッキング）へ置換された。
    `RunArena` は領域の所有と flush-before-destroy だけを持つ。
- 根拠資料: recon §2/§3/§4（不変条件 14 項目は全て先行実験プロジェクト（以下プロトタイプ）
  の実測障害由来）

## 決定

### 形状

- **全ノードの出力 shape を実行前に確定させる（静的形状）**。データ依存出力形状
  （nonzero 等）は非対応。将来対応するなら「固定上限 + マスク」を個別 ADR で裁定する
  （実行モデル全体に波及するため安易に変えない）。

### GPU 実行

- dispatch 列は**時間予算ベースで複数 command buffer に分割して submit**
  （TDR 2 秒 / Chromium watchdog 対策）。チャンクサイズ 0 は構築時に拒否。
  **改訂（2026-08-04・ADR 0032）**: 旧「チャンク時間は前チャンク完了時刻からの差分」は廃止 —
  submit ごとの `onSubmittedWorkDone` は Deno で CPU/GPU を直列化するため、計測は
  flush の待ち 1 回へ集約した（窓の定義は下の④）。
- 適応制御の不変条件（M1-P2 で改訂 — 詳細は `packages/runtime/src/gpu/submit.ts` の doc）:
  ① **成長は実測の裏付けにのみ基づく**（実測 0 = タイマ分解能未満は「速い」ではなく
  「情報が無い」— 裏付けが無い間は初期チャンクサイズで据え置く）②積む前に判定して
  **1 チャンクの推定時間が予算を超えない**ようにする（超えてから縮めない）③推定の単位は
  dispatch 数ではなく **workgroup 数**（記号次元のグラフでは入力長で 1 dispatch の重さが
  変わるため、dispatch 数で測ると短い入力で得た推定のまま長い入力を積む）④個々のチャンク
  への時間帰属は信用しない（重なった submit の完了通知はほぼ同時に届き、先頭 1 本に全時間が
  乗る）— 推定は窓の実測 ÷ 合計 workgroup 数で取る。**改訂（2026-08-04・ADR 0032）**:
  窓 = 「窓で最初に submit した時刻 → flush の `onSubmittedWorkDone` 解決時刻」（旧
  「キューが空になるまで」から flush 単位へ）。窓はホスト側エンコード時間を内側に含み
  推定は過大 = チャンク縮小向き = TDR 安全側。仕事量 0 の窓も「情報なし」扱い
  （0 除算 → Infinity → minChunkSize 永久張り付きの防止）。
- **device.lost はランタイム一級イベント**。無視すると mapAsync が永久ハングする。
  device / PipelineCache / スケジューラは `device.destroy()` 後に再構築できる構造にする
  （VRAM を返すのは device.destroy() のみ — buffer.destroy() は 1 バイトも返さない）。
- `requiredLimits` は compute 系（workgroup storage / invocations / workgroupSize X,Y,Z）
  まで明示要求し、取得後に検証する。`pushErrorScope('validation')` を常設する
  （無効パイプラインは throw せず dispatch no-op → 出力全 0 の沈黙故障）。
- codegen は **決定性（同一キー → バイト単位同一 WGSL）** を不変条件とし
  スナップショットで固定。elementwise と行 reduce 族は grid-stride 前提。

### バッファ管理

- per-run アリーナ（サイズ別プール + 最終消費者解放）。不変条件: ① release はノード境界
  のみ ② 出力 alloc は dispatch エンコード前 ③ useCounts は `node.ins` の厳密延べ計数
  ④ `queue.writeBuffer` 対象はプール外 ⑤ **full-write: 全ノードは出力バッファの全バイトを
  書く** ⑥ 破棄・失敗経路では必ず flush してから destroy。中間値の readback は拒否する。
- ⑤ の改訂（2026-08-02 / ADR 0014）: 当初は「全書きしない op は新品を要求」（プールを迂回）
  だったが、不変条件がカーネル実装の知識に依存する（アリーナ側が op 種別を知る必要が生じる）
  ため却下し、**カーネル側に全域書きを課す**形へ反転した。プール再利用バッファはゼロ初期化
  されないので、ゼロ埋めが要る op（pad）は範囲外にも 0 を**書く**。1 ノードが複数 dispatch を
  出す形（cat）はノード単位で全域を覆えばよく、被覆は呼び出し側が offset の総和と出力の軸長で
  突き合わせる。固定はフォールト注入（再利用バッファへ毒値 → 実行 → 毒値が残らないこと —
  `packages/runtime/tests/gpu_full_write_test.ts`）。

### 規約

- **fail loudly 全域**: 未対応 op / 属性・格納 dtype の流出・未束縛シンボル・broadcast
  不可・負 pad は黙って近似せず例外。シンボル束縛の参照は `Object.hasOwn` 経由。
- **初期化は明示 async ステージ**（プロトタイプはコンストラクタ同期ループが 28.5 秒の床だった）。
  重み取得と GPU アップロードをパイプライン化できる入口（チャンク逐次投入）を持つ。
- グラフ最適化パス（恒等 expand の別名化・permute 連鎖畳み込み・融合）を一級機能とする。
  融合は最適化のみで、正しさは常に非融合経路が担保する。

## 未決（個別 ADR 待ち）

- KV キャッシュの IR/実行表現（静的最大長 vs prefill/decode 分割）— プロトタイプでも未決。
- 時間予算の既定値の根拠づけ（プロトタイプは 100ms・安全率 0.5 の経験値）。
- メモリプランの発展形（厳密 liveness / 静的アリーナ計画）。
