# 0032: submit スケジューリングの非直列化（onSWD 廃止）と GPU 時間診断の明示 opt-in 化

- Status: accepted（2026-08-04 ユーザー承認「それぞれ推奨案で承認・完走前提」）
- Date: 2026-08-04
- 関連: ADR 0004（適応制御 — **決定④「窓 = キューが空になるまで」を本 ADR が改訂**）/
  0021（GPU 時間診断 — **決定 1「undefined = 自動判定」を本 ADR が改訂**。帰結節が予告した
  「問題が出れば既定を false に倒す再裁定」の実行）/ 0030・0031（「非 GPU 固定費」の記録に
  但し書き — 実体は本 ADR の Context）/ recon =
  [2026-08-04-host-overhead-recon.md](../research/2026-08-04-host-overhead-recon.md)
  （帰属の実測・単離実験・案 1〜3 の設計）

## Context

1024px turbo の「非 GPU 固定費 695〜704ms/step」の帰属 recon で、実体が 3 つに分解された:

1. **計測装置代 370〜375ms/step（解像度非依存）**: `gpuTiming` の自動判定既定により
   デモ・ベンチが常に計測有効で走り、1 dispatch = 1 pass（ADR 0021 決定 2）の
   `encoder.finish()` 代価が乗っていた。単価は **pass 数に線形 × その device で作られた
   累積バッファ数に線形**（micro 実測 — 生存数ではなく累積生成数。機序は wgpu 内部の
   推測、相関は実測）で、params バッファを毎 dispatch 使い捨てる現行構造では
   **step 番号とともに単調増加する**（step 2→10 で非 GPU 470→706ms。熱ではない —
   計測 off では平坦）。
2. **真のホストエンコード代 ~302ms/step**（params 毎 dispatch 新品 139ms・bind group
   52ms・JS 残差 81ms 等 — recon §3.1 の内訳表が正本）。
3. **構造の根本原因: submit ごとの `queue.onSubmittedWorkDone()`（fire-and-forget）が
   Deno/wgpu では同期部分で「そこまでの全作業完了」までホストをブロック**し、CPU
   エンコードと GPU 実行が完全直列化していた（壁時計 = GPU + ホストの**加算**。素の
   WebGPU 単離実験で「呼ぶと wall=gpu+host / 呼ばないと wall=max(gpu,host)」— recon
   §4.1・メイン再実行でも一致）。ブラウザの onSWD は非ブロッキングで、この直列化は
   Deno でだけ起きる。

errorScope（0.03ms/run）・GC（mutator 0.999）・await 連鎖（残差 81ms）は無罪と確定。

## Decision

1. **submit ごとの `onSubmittedWorkDone` を廃止する**（packages/runtime/src/gpu/submit.ts）。呼ぶのは
   `flush()` の 1 回だけ — この待ちは flush-before-destroy のために元から必要で、
   計測のための追加ブロックはゼロになる。CPU エンコードは GPU 実行の裏へ重なる。
   **再改訂（2026-08-13・ADR [0054](0054-resident-loop-and-fence.md) 決定 6）**: 通常 run
   （gpuTiming OFF かつグラフ出力 ≥1）の完了構造は **`mapAsync` が唯一の fence** へ再設計され、
   flush の onSWD と arena.destroy 後の再 flush は削除された。**スケジューラ側の窓設計
   （決定 2）は引き続き有効** — 変わったのは run 完了の待ち方だけ。
2. **適応制御の観測窓を flush 単位に再定義する**: 窓 = 「窓で最初に submit した時刻 →
   flush の onSWD 解決時刻」、推定 = 実測 ÷ 窓の合計 workgroup 数。ADR 0004 の
   不変条件①〜③は維持（実測 0 **と仕事量 0** は情報なし = 更新しない〈仕事量 0 は
   0 除算 → Infinity → minChunkSize 永久張り付きの防止〉・積む前判定は無変更・
   workgroup 単位）。新推定は**ホスト側のエンコード時間を内側に含むため GPU 実時間より
   過大** = チャンクが縮む向き = TDR 安全側。歯止めは minChunkSize / maxChunkSize。
3. **`gpuTiming` の既定（undefined）を「要求しない」へ**（ADR 0021 決定 1 の改訂・
   shaderF16 と同一規律）。`true` = 必須（不在は fail loudly）は不変・`false` は既定と
   同値の明示形。理由: 自動判定では計測の代価（1 dispatch = 1 pass）が**アダプタの
   能力で既定経路に乗り**、「計測すると遅くなる」状態が perf の基準になる。診断は
   明示的に払うもの。
4. **デモは `--gpu-timing`（既定 off）**で従来の内訳 JSON（gpuTimingByGraph）を明示
   opt-in に。**素の時間を測る run と内訳を採る run は分ける**（ベンチ規約）。
5. **`SubmitStats.pendingMeasurements` は削除**（fire-and-forget の未着件数という概念が
   消滅。残すと常に 0 の派生不能フィールド。公開面の破壊的変更 — 未リリースにつき
   シムなし）。

## 実測（2026-08-04・実装後・クールダウン規約適用 = アイドル基準 46°C・毎 run 前 +5°C 以下）

運用形（turbo `--dit i8 --linear-compute i8a8 --attention-compute i8a8
--attention-score-storage f16`・seed 42・10 step 平均）:

| 指標            | 旧既定（計測 on・旧スケジューラ） |         新既定（本 ADR・×2 run） |           差 |
| --------------- | --------------------------------: | -------------------------------: | -----------: |
| 1024px 壁/step  |                    2,131〜2,299ms | **1,502〜1,512ms**（定常 1,480） | **−29〜35%** |
| 1024px 生成全体 |                       29.1〜31.0s |                  **22.5〜22.6s** | **−23〜27%** |
| 512px 壁/step   |                           971.6ms |  **431〜438ms**（定常 412〜415） |     **−55%** |
| 512px 生成全体  |                             16.0s |                  **10.3〜10.7s** | **−33〜36%** |

- 旧値は同一構成・同一 seed の排他 A/B 実測（当時の既定 = 計測 on・クールダウンなし）。
  比較は「ユーザーから見た既定どうし」で、装置代（−375ms）と直列化解消の分解は recon の
  帰属表とスタブ実験（非 GPU 705.6→161.9ms）が正本。
- **PNG sha256 門 2 系列とも実行前後で完全一致**（ビット同一 — WGSL・キー無変更の裏取り）。
- `--gpu-timing` 動作確認: 内訳は 4 グラフ全てに出力・off run の JSON に欄は現れない。
  timing on の 512px は step 457→691ms と単調増加が残る（装置代の性質そのもの —
  「素の時間と内訳採取は別 run」規約の根拠）。
- 適応制御は定常 submit 29/step（1024px）・7/step（512px）・chunk 1024 で収束。
  実 GPU プローブでは run 1 の窓 1 本で推定が付き run 2 以降 1 submit/run
  （推定 1.69e-4 → 1.67e-4 ms/wg）。
- verify **668 passed / 0 failed / 4 ignored**（メイン自己実測）。故障注入 6 件 +
  検出限界の実証 1 件（実装波レポート）。

## 検出限界・知見（本タスクの新記録）

1. **本 ADR の退行（per-submit onSWD の復活）は数値網で原理的に検出不能** — 出力も統計も
   1 ビットも変わらず、遅くなるだけ。検出器は `onSubmittedWorkDone` の**呼び出し回数を
   数える結線検査 1 本**（packages/runtime/tests/gpu_submit_test.ts の FakeGpu.calls — MUST を doc 化）。
2. **キー検査（lastRunTiming の entries を見る形）は計測が無効だと黙って空振りする** —
   既定 off 化に伴い、キー期待値を壊しても素の `acquireGpu()` では緑のまま通ることを実証
   （故障注入 F7）。packages/runtime/tests/helpers/gpu.ts の `TIMING_ACQUIRE_OPTIONS` 経由が MUST。
3. **`encoder.finish()` の単価は「累積作成バッファ数」に比例し destroy では下がらない**
   （micro 実測 +5µs/pass/1,000 本）— params キャッシュ（recon 案 2・保留枠）が装置代の
   成長も止める根拠。
4. 計測経路（1 dispatch = 1 pass）のテスト被覆は既定 off 化で「偶然の被覆」を失った —
   明示 `gpuTiming: true` のテスト（timing / conv2d parity / キー検査 15 ケース）だけが
   踏む。新カーネル追加時はこの経路の被覆を意識すること。

## Consequences

- **壁時計 = max(GPU, ホスト) に近づいた**: 以後の dispatch 増（例: attention a8 の
  +224/step）が壁時計へ直撃しなくなり、recon（intermediate-f16-design §3.4③）の
  「変換 dispatch 挿入は非 GPU を悪化させる」制約も緩む。ホスト律速側（512px 以下・
  SBV2 / DeBERTa）には案 2（params / bind group キャッシュ・−139〜191ms/step 相当）が
  保留枠として残る。
- **適応制御の粒度が run 単位になった**（従来はキューが空になるたび）。初回 run は
  initialChunkSize のまま走り切る（保守的側 = TDR 安全）。
- **ADR 0030/0031 の「非 GPU 固定費 ~350 / 695ms/step」は装置代込み・step 依存の値**
  だった（本 ADR の Context）。両 ADR に但し書きを同梱。以後の perf 実測は「既定
  （計測 off）で素の時間・内訳が要る時だけ `--gpu-timing` の別 run + 毎 run 前
  クールダウン」を規約とする。
- msPerWorkgroup はホスト込みの上限推定になった。GPU timestamp が有効なときにそちらを
  推定源へ使う改良は保留枠（recon §5 案 1 の別案）。
