# 0021 — GPU 時間診断（timestamp-query・op 別内訳）

- Status: accepted（2026-08-03・ユーザー裁定 = 案 A + 三値 auto 既定）
- 目的: perf マイルストーンの前提となる **op 別の GPU 実時間内訳**。壁時計の段別計測
  （デモ側 JSON）ではチャンク・op が混在し、最適化対象の順位づけができない。
- 前提の実測: submit は「1 チャンク = 1 pass」（packages/runtime/src/gpu/submit.ts）で、WebGPU の
  timestamp は pass 境界（`timestampWrites`）にしか置けない。pass 内 per-dispatch 計時は
  wgpu ネイティブ拡張であり Deno / Chrome の標準 WebGPU には無い。

## 決定

1. **opt-in は `acquireGpu` のオプション**（feature は device 作成時にしか要求できない）。
   `gpuTiming?: boolean` の三値:
   - `undefined`（既定）= **自動判定**: アダプタが `timestamp-query` を持てば要求して計測、
     無ければ計測しない（無効は `diagnostics()` で観測可能 — 黙って近似しない）。
     **改訂（2026-08-04・ADR 0032）**: 既定は「要求しない」へ変更（自動判定廃止・
     shaderF16 と同一規律）。計測の装置代（1 dispatch = 1 pass の finish 代 370〜375ms/step）
     がアダプタ次第で既定経路に乗り perf の基準が歪むため — 帰結節が予告した再裁定の実行。
   - `true` = 必須: feature が無ければ **acquireGpu が fail loudly**。
   - `false` = 無効: feature があっても**要求しない**（既定経路を 1 バイトも変えたくない
     利用者・ベンチの対照用）。
2. 計測時のみ **1 dispatch = 1 pass** で encode し、pass の begin/end を chunk ごとの
   querySet（容量 2 × チャンク dispatch 数 ≤ 2,048 — maxChunkSize 1024 と WebGPU の
   querySet 上限 4,096 の内側）に書き、`resolveQuerySet` → readback で回収する。
   同一 pass 内の連続 dispatch には元々暗黙の依存順序（storage の可視性保証）があるため、
   pass 分割で実行意味論は変わらない。
3. 集計は**パイプラインキー別**（ns 合計・dispatch 数・workgroup 数）。キーは op 種 +
   変種（`:wi8` 等）を既に一意識別する語彙なので、集計用の新しい命名は導入しない。
   `Session.diagnostics()` に **直近 run の表**として載せる（lastRun 系と同じ寿命）。
4. requiredFeatures 全廃方針（M0.1）は「**既定で**は何も要求しない」に読み替える —
   `false` 指定と feature 無し環境では従来どおり requiredFeatures は空。
   `enable f16` など**シェーダ側**の feature 依存ゼロは不変。
   **改訂（2026-08-04・ADR 0028）**: 後半の「シェーダ側の feature 依存ゼロ」は
   「**既定経路では**不変」に狭める — f16 計算変種（opt-in・`acquireGpu({shaderF16: true})`
   必須・実走カナリア付き）だけが `enable f16` を出す。既定の f32 経路の生成物は
   引き続き feature 依存ゼロで、1 バイトも変わっていない（スナップショットが検出器）。

## 検証

- 実 GPU: 診断の dispatch 数合計が既存 SubmitStats.dispatchCount と一致する相互検算・
  全エントリ ns ≥ 0・キー集合が実行された pipeline のキー集合と一致・feature 不在時の
  三値それぞれの挙動（true = throw の結線 / undefined = 無効を観測 / false = 非要求を
  device.features で観測）。
- 計測モードは実行結果を変えない: 同一グラフを gpuTiming on/off で走らせ出力バイト一致。

## 帰結・トレードオフ

- 自動判定既定なので、feature を持つ環境では常時 op 別内訳が手に入る（ADR 0006 の
  常設診断と同じ思想）。計測コストは encode の pass 分割 + チャンクあたり ≤16KB の
  readback で、実測対象（秒級の run）に対しては無視できる見込み — 実測で裏取りし、
  問題が出れば既定を `false` に倒す再裁定を行う。
  **改訂（2026-08-04・ADR 0032）**: 「無視できる見込み」は反証された — 装置代は実測
  370〜375ms/step（解像度非依存・作成済みバッファ累計で単価成長）で、既定を「要求しない」
  へ倒した。内訳が要るときは `gpuTiming: true`（デモは `--gpu-timing`）の別 run で採る。
- タイムスタンプの分解能はブラウザ実装が量子化しうる（Chrome の緩和策）。集計は数百〜
  数千 dispatch の合計なので影響は薄いが、単発 dispatch の絶対値は参考値として扱う。
- 却下案: B = チャンク粒度計測（op 混在で内訳にならない）/ C = CPU 時刻の細分化
  （onSubmittedWorkDone は累積完了 — ADR 0004 の既知の罠）。
