# 0022 — GEMM 3 op のレジスタ 4×4 + vec4 置換

- Status: accepted（2026-08-03・ユーザー承認済みの 6 裁定を含む）
- 対象: `matmul` / `bmm` / `linear`（packages/runtime/src/kernels/gemm.ts + 薄い呼び出し面 3 ファイル）
- 需要の実測: DiT の GPU 実時間内訳（ADR 0021 の診断）が linear 89%（512px・wf16）/
  75%（1024px・wi8）・bmm 9% / 23%。プロトタイプ実測 3.07×
  （[research/2026-08-03-demo-w8-perf-recon.md](../research/2026-08-03-demo-w8-perf-recon.md)）。
  設計書 = [research/2026-08-03-gemm-design.md](../research/2026-08-03-gemm-design.md)。

## 決定

1. **共通骨格**: 64×64 出力タイル・16×16=256 スレッド・1 スレッド 4×4
   （`acc: array<vec4<f32>,4>`）・K タイル 16・共有 8KB の 5 断片差し込みテンプレートを
   `packages/runtime/src/kernels/gemm.ts` に 1 本だけ置き、3 op が共有する。旧実装の「共通化するな」MUST は
   目的（既存生成バイト列の保護）が 3 本同時の総取り替えで消えたため文言ごと書き換えた —
   **内積ループの正本 1 箇所化が「3 op の縮約順序同一」を機械的に守る**。
2. **キーは v2 + 形状由来の v4 フラグ**（`gemmUsesVec4(k,n) = k%4==0 && n%4==0`）。
   端数形状は**同一レジスタ構造のスカラ変種**（実測系譜 2.50×）へ落ちる。キーが形状の
   関数になるのは `elementwiseKey` の `r${rank}` と同じ語彙で、決定性（同一キー ⇔
   同一バイト列）は崩れない。パディング案は不変条件 3 本（full-write・別名化・プール寸法）に
   波及するため却下、16×16 残置はスカラ変種の下位互換のため却下。
3. **縮約順序は旧カーネルと完全一致**（K 昇順・タイル幅 16・外側 t / 内側 kk とも昇順）。
   MUST: これは数値契約の土台であり、**縮約順序を変える将来の変種（DP4a・shader-f16 等）は
   tolerance 再導出とセットでしか入れない**。
4. **重み格納の quad 読みは weight-storage.ts の opt-in 拡張**（`dequant4` / `weightRead4`）。
   既定 false で既存 4 op（conv1d/conv2d/conv_transpose1d/embedding）の生成物は 1 バイトも
   動かない（fixture 6 本のスナップショットが検出器）。i8 は `vec4<f32>(unpack4xI8(w)) * s`
   の**成分ごと f32 乗算**でスカラ経路と同一の丸め — **ADR 0019 の「scale は要素ごと」は
   vec4 化でも維持され、同 ADR の改訂は不要**。f16 は u32 2 語で 4 要素（平坦添字が 4 の
   倍数であることに依存 — ADR 0018 の偶奇の罠とは別物で、スカラ経路の検出器は温存）。
5. **bmm の uniform は 3 語 `{m,n,k}`**（バッチは `wid.z` から導出。旧 4 語の `batch` は
   WGSL が一度も読まない死フィールドだった）。

## 検証（全て実測済み・2026-08-03）

- **ビット同一の実証**: 置換前後で Anima デモの PNG sha256 が **512px/wf16・1024px/wi8 の
  2 系列とも完全一致**（10 step DiT + VAE decode の全経路・v4 経路）。加えて全 E2E
  （tiny 25 / DeBERTa / SBV2 5 系列 / Anima f32・f16・i8・1024・turbo 約 110 比較）の
  atol=rtol=0 実測 maxAbs が置換前の記録値と記載桁まで完全一致 — **tolerance は 1 つも
  変えていない**。NOTE: 後者は記載桁（3〜4 桁）までの一致であり全要素ビット一致の直接証明
  ではない（直接証明は PNG の 2 系列）。
- **性能**: DiT GPU 実時間/step が 512px 3,741→946ms（**3.95×**）・1024px
  22,153→6,071ms（**3.65×**）。壁時計/step 2.60× / 3.90×。テストスイート所要 5m47s→3m24s。
  事前予測 2.95×（下限 1.8×）を上回った。
- **故障注入 10 件**（バッチ base の quad 単位・端タイル quad ガード・K 端数 0 埋め・
  転置撒き添字・f16 2 語目オフセット・f16 偶奇反転・i8 scale 軸・v4 判定・タイル辺定数・
  A/B 相殺の予備調査）で対応テストの赤→復元を実測。**検出限界 3 件を記録**:
  ①K 端数 0 埋めは A/B 片側だけの退行を検出できない（相手側の 0 埋めが積を相殺）
  ②matmul 単独の K 端数退行は現行ケースで捕まらない ③i8 scale 軸の取り違えは行タイル
  2 枚以上のケースのみが検出器（`arow == wcol` に縮退する形では値に出ない）。
- **full-write**: GEMM 3 op × v4/スカラの毒値注入 6 ケースを常設
  （packages/runtime/tests/gpu_full_write_test.ts）。縮退ハーネス（grid-stride）は**カテゴリ違いで対象外**
  （タイル系の安全網は tiledWorkgroups の fail loudly + full-write テスト）。

## 変種の踏み分け（IR からの机上 census・実測束縛値で確認）

- Anima DiT は全 8 コンテナの GEMM 566 ノードが**全て v4**。VAE の bmm 2 本も v4。
- SBV2 front/flow/voice は**形状固定の scalar bmm**（k=9 / n=9）を恒常的に踏む + P/T 依存の
  実行時切替。DeBERTa / Anima text 系は記号次元（T/Tsrc/Ttgt）の 4 の倍数性で切替。
- **matmul op は実モデルに 1 ノードも存在しない**(全て linear / bmm に落ちる)。
- カバレッジの穴（記録): matmul スカラ変種と f32/f16 格納の linear スカラ変種は E2E に
  実モデル経路が無く、ユニットテスト（gpu_ops_test 等）のみが検出器。

## 帰結

- 512px 生成 26s（61.4s→）・1024px 生成 87s（~290s→）。1024 では未置換の VAE conv2d
  （15.7s・96%）が相対的に浮上 — perf 次段（融合 attention / conv2d）の入力。
- `acc` の動的添字ループはドライバの展開に依存する（実測系譜の形のまま採用）。他環境で
  性能が出ない場合の第 1 候補は codegen 時の 4 本展開（設計書 §2.7）。

> 追記（2026-08-10）: 上の予言は NVIDIA でも成立していた — 動的添字はローカルメモリ退避で
> 律速し、i8a8 系は codegen 時展開へ移行済み（linear = `4b15ec2`・attention = `3f417dc`、
> 各 ×1.35〜1.45）。さらに **i8a8 GEMM 族はタイル幾何を f32 骨格から独立させてパラメタ化**した
> （`src/kernels/i8a8-geometry.ts`・実測最良の既定 = linear/QK M128N64 r8×8 wg8×16 / PV
> M64N128）。根拠は整数縮約の順序非依存（幾何を変えても出力ビット不変 —
> [research/2026-08-10-kernel-variant-sweep.md](../research/2026-08-10-kernel-variant-sweep.md)）。
> **決定 3 の MUST（縮約順は数値契約・変種は tolerance 再導出とセット）は f32/f16 骨格
> （gemm.ts）に引き続き適用**され、そちらの幾何は本 ADR のまま動いていない。
