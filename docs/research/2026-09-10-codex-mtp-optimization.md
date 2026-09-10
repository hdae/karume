# レビュー指摘の再検証と修正（2026-09-10）

この文書は 2026-09-10 時点の調査・実測スナップショットであり、恒久的な作業状態は [backlog](../backlog.md) を正本とする。

## 対象と環境

利用者が指定した `.claude/reviews/2026-09-11_chatgpt-reviews/` の 3 レビューを主担当が直接確認した。
ディレクトリ名の日付と実行日付は区別する。序盤のみ GPT-5.6 Sol / Terra に 9 月 6 日および直近レビューの棚卸しを委譲し、以後の調査・修正・検証は主担当が行った。
開始点は `4dca96a`、作業ブランチは `codex/review-and-fix`。最新の利用者指示を優先し、旧作業指示の `codex/mtp-optimization` へは切り替えていない。他ブランチへの変更はない。

実測は Linux / Deno 2.9.6 / RTX 3080 Ti（driver 610.57.04）で実施。
Python は `tools/exporter` / `tools/export-recipes` の `uv run --locked` 環境（Python 3.14.6 / PyTorch 2.13.0+cpu）。
Apple M2 とブラウザは未実測。外部リポジトリのクローンや外部リポジトリ由来の実装の複製は行っていない。

生データと実行スクリプトは新規の [出力ディレクトリ](../../outputs/bench/karume/2026-09-10_review-and-fix/) に保存した。
このリンク先と `.claude/reviews/` は git 追跡外。再現用スクリプト・ログは同ディレクトリ内にあり、既存のモデル資産を上書きしていない。

## 9 月 11 日レビューの対応

ID はディレクトリ内のレビュー番号で区別する。同じ原因の指摘はまとめて扱う。

| 指摘         | 確認した原因と変更                                                                                                                                       | 検証                                                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1/B01、3/F03 | native async generator は未開始の return / throw で finally を実行しない。内部の共通ラッパーで未開始の終了を通知し、done・busy・仮追加発話を決着させる   | 未開始 return / throw、二重終了、開始直後の next / return の順序、KV 保持、GPU run 未実行                                         |
| 1/B02        | draft の onRun が cycle の計測区間内に入り自己採算ゲートを変えていた。即時通知は保ち、同期 callback の所要時間を wallMs から引く                         | callback 費用 0 / 5 / 50 ms の偽時計で token・停止・ゲートイベントが同じ                                                          |
| 1/B03、3/F04 | done が後始末より前に成功していた。共通の終了処理で cleanup とリース返却後に通知する                                                                     | 後始末を保留した間 done も保留。単独失敗・本体との AggregateError が done と iterable で同じ。done 直後に次ターン開始可能         |
| 1/B04、3/V02 | dispose 後に継続した PLE 読み込みがキャッシュへ再登録していた。dispose 済みなら所有者の表へ戻さない                                                      | open / readAll / range の各待機点で dispose、先行 gather は完了し resident 件数・bytes は 0                                       |
| 1/B05、2/C03 | Irodori の最初の cleanup 失敗が後続解放を打ち切り、本体失敗も失われた。既存 disposeSteps で batch → Session → ResidentTensor の順と全段の試行を維持      | 実 GPU の常駐ループへ多重故障を注入。batch 1 本、Session 3 本、resident 7 本を順に解放し、5 件の原因を保持                        |
| 1/P01        | draft が空でも履歴全体をコピーしていた。plain step では元の readonly history を sampler に渡す                                                           | 履歴への要素アクセスを拒否する Proxy と同一参照の検査                                                                             |
| 2/C01、2/C02 | conv bias / i8 scale / W4A8 group scale が端チャネルを無条件に読んでいた。読出し前に添字を有効な最終チャネルへ制限                                       | codegen の正確な文字列・snapshot と GPU 端数形状。既存の数値許容誤差やビット同一条件は維持                                        |
| 3/F01        | 形状依存 reshape は prefix と可換とは限らず、Tmax / Tmax−1 だけでは反例を見逃す。記号 view は単位軸の追加・削除だけを fold し、他は実行時 reshape へ残す | 修正前に添付反例の不一致を再現。動的軸 3 位置 × T=2..16 の回帰。詳細は [ADR 0010](../decisions/0010-symbolic-constant-folding.md) |
| 3/F02        | 5 / 9 のマスク実評価だけでは T≤4 の全マスク行を見逃す。記号依存が残るマスクは safe_softmax を維持                                                        | 修正前にゼロ→NaN を再現。回帰テストと [ADR 0044](../decisions/0044-runtime-attention-mask.md)                                     |
| 3/F05        | 配列だけの freeze はメッセージを共有する。各発話をコピーして freeze、overflow の返却物も内部用にコピー                                                   | snapshot・overflow の入力変更を拒否し、overflow が返した発話を後で変えても内部履歴が変わらない                                    |
| 3/F06        | 辞書を追加のたびに全コピーしていた。fetchAssets と Gemma fromAssets の辞書を一度で組み立てる                                                             | `__proto__` 等の own property、列挙順、参照共有、取得の重複排除と下記ベンチ                                                       |

GPU の範囲外アクセスは、最終 store が無効 lane を捨てるだけでは安全といえない。
[WGSL の範囲外アクセス規則](https://www.w3.org/TR/WGSL/#out-of-bounds-access) を確認し、値を読んだ後の select ではなく添字を先に制限した。
有効 lane の読み値・積和順・group flush の丸め位置は変えていない。新しい runtime カーネルや公開 API は追加していない。

## Exporter の反例

修正前の `karume-review-export-counterexamples.log` では、公開時の構造検証を通過しても F01 が T=2,6,8、F02 が T=2,3,4 で参照と異なった。
この段階の比較器はレビュー添付の小さな CPU IR 評価器であり、GPU の結果とは区別する。

修正後の `export-counterexamples.py` は 2 モデルを公開形へ保存し、各 T=2..16 の PyTorch 参照値を `F01/reference.json` / `F02/reference.json` に保存する。
F01 の op は `sym_prefix_slice → reshape → add`、F02 は `sym_prefix_slice → add → safe_softmax` となった。
`export-gpu.ts` がこの公開形を実 runtime で検査した。全 30 条件で非有限値なし、PyTorch 参照との最大絶対誤差 0（`export-gpu.jsonl`）。

F01 の修正は記号 reshape の構造的な制限であり、allowlist 内の任意の op が prefix と可換であるという一般的証明ではない。
安全な reshape の一部も実行時へ残す保守的な変更である。既存の静的 fold・単位軸の変更・レシピ全体の回帰を確認した。
既に公開済みのモデル資産を再 export / 再公開する作業は行っていない。

## 資産表構築の実測

`hub-bench.ts` が実際の fetchAssets を呼ぶ。メモリ上の localDirectory、1 byte の資産、各件数で warmup 3 回と計測 9 回。
毎回、結果のキー数と読み出し回数を確認する。修正前後の JSONL は `hub-before.jsonl` / `hub-after.jsonl`。

| 資産数 | 修正前中央値 ms | 修正後中央値 ms |
| ------ | --------------: | --------------: |
| 100    |          0.2533 |          0.2159 |
| 1,000  |         33.3914 |          0.9012 |
| 3,000  |        843.6710 |          2.1189 |

3,000 件の min–max は修正前 841.73–853.41 ms、修正後 2.07–2.31 ms。
同期の表構築の二乗コストを除けた。ネットワーク・大きなファイル・実モデルの cold load の倍率へは外挿しない。
GPU 端チャネル保護の前後比較は以下のとおり。

## GPU 端チャネル保護の初回実測と一時停止

`gpu-guards-bench.ts` は開始点の Conv1d WGSL snapshot と修正後の生成コードを同一 GPU / 入力で交互に実行する。
f32 / i8 の vec4 経路、各条件 warmup 3 round、計測 9 round、1 sample は同じ pass 内の 32 dispatch。
GPU timestamp の差を dispatch 数で割る。コンパイル・転送・readback は計測外。
`gpu-guards.jsonl` に 108 sample。各条件の前後で出力 u32 の全要素一致も検査した。

| 格納 | 入力チャネル / 出力チャネル / 長さ | 修正前中央値 µs | 修正後中央値 µs |   変化 |
| ---- | ---------------------------------- | --------------: | --------------: | -----: |
| f32  | 8 / 5 / 20                         |           8.978 |           9.459 | +5.36% |
| f32  | 64 / 128 / 1024                    |          52.681 |          53.738 | +2.01% |
| f32  | 64 / 129 / 1024                    |          52.778 |          53.850 | +2.03% |
| i8   | 8 / 5 / 20                         |           9.027 |           8.595 | −4.79% |
| i8   | 64 / 128 / 1024                    |          51.590 |          48.742 | −5.52% |
| i8   | 64 / 129 / 1024                    |          52.134 |          49.236 | −5.56% |

全ケース batch=1 / kernel=3 / stride=1 / padding=1 / dilation=1 / groups=1。
**f32 の単体性能低下を検出したため、利用者の「想定外の問題では停止」の指示に従って追加の調査・修正・計測を停止した。**
i8 の改善を f32 の低下と相殺して採用判断しない。モデル全体の速度低下率を示す数値でもない。
今回の計測は Conv1d の限定形状であり、Conv2d / linear i8 / W4A8 の性能へ外挿できない。

初回は安全な添字保護を差分に残した未コミット状態で停止した。その後、利用者から保護のコストは想定内として改善調査の続行が承認された。
再開後は bias 初期化の WGSL だけを変え、積和・store を固定して比較した。採用結果は次節に記す。
範囲外読みへ戻したり、数値条件を弱めたりして性能を回復させない。

## 再開後の調査と採用

追加のスクリプト・WGSL・生データは [guard-cost 出力](../../outputs/bench/karume/2026-09-10_review-guard-cost/) に保存した。
各 submit の先頭に無計測の 128 dispatch を置き、アイドルからの立ち上がりを計測区間から外す。
同じ GPU / 入力の比較順を round ごとに回転・反転する。絶対時間は別走行間で比較せず、同一走行の前後で判断する。

- `candidates.jsonl`（450 sample）: 行ごとの read の共通化、0 番への select、個別の条件分岐は min より良くなかった。
- `bounds.jsonl`（450 sample）: arrayLength による物理境界の保護もほぼ改善なし。完全な行ブロック・タイルで読出しを分ける案は f32 で有効だが i8 は悪化。
- `matrix.jsonl`（1,944 sample）: Conv1d / Conv2d × f32 / f16 / i8 × scalar / vec4 × 32 / 64 行タイル × 出力チャネル 5 / 64 / 65 の 72 条件。
  全候補で u32 一致。行ブロック・タイル分岐を全条件へ一律に適用すると悪化する。今回の f32 Conv1d 問題には vec4・64 行タイルだけが安定して有効だった。
- `tile-count.jsonl`（180 sample）: 完全なタイルの判定を `wid.y < dims.m / 64u` にすると、上端を乗算・加算して比較する式より速かった。
  この式は加算 overflow も避けられる。`wid.y < floor(Cout/64)` なら `64·wid.y + lid.y·8 + row < Cout` が全 128 invocation・row=0..7 で成立する。
  端タイルでは従来どおり min で添字を保護する。どちらの分岐も積和より前に bias を acc へ代入し、後続 barrier の到達条件を変えない。

**採用範囲は f32 Conv1d / vec4 / 64 行タイルに限定**した。新しい op・dispatch・公開 API・設定は追加していない。
ほかの格納形式・Conv2d・32 行タイル・scalar は最初の保護実装を維持する。
f16 の一部にも有利な候補はあったが、今回の f32 低下への対処と分け、適用を広げていない。
命令・レジスタ割当単位の帰属は未確認である。bias 初期化以外の WGSL を固定した比較なので、この表現変更による差とまでは切り分けた。

最終の `final-bench.ts` / `final.jsonl` は 18 条件（f32 / i8 × 9 チャネル数）× 3 実装 × 15 round = 810 sample。
各 sample は GPU timestamp / 32 dispatch。変更前・最初の保護・最終実装の全出力を u32 で突合した。
以下は f32 の中央値（µs/dispatch）。いずれも vec4 / 64 行タイルを明示指定している。

| Cout | 保護前 | 最初の保護 | 最終実装 | 最初の保護に対する変化 |
| ---- | -----: | ---------: | -------: | ---------------------: |
| 1    |  7.911 |      8.351 |    7.491 |                -10.30% |
| 5    |  8.106 |      8.536 |    7.679 |                -10.04% |
| 32   | 49.690 |     50.697 |   47.227 |                 -6.84% |
| 33   | 49.778 |     50.706 |   47.253 |                 -6.81% |
| 63   | 49.450 |     50.452 |   46.702 |                 -7.43% |
| 64   | 48.299 |     49.254 |   45.421 |                 -7.78% |
| 65   | 48.427 |     49.403 |   45.678 |                 -7.54% |
| 128  | 50.394 |     51.383 |   47.483 |                 -7.59% |
| 129  | 50.065 |     51.078 |   47.252 |                 -7.49% |

Cout≤5 は Cin=8 / L=20、それ以外は Cin=64 / L=1024。batch=1 / kernel=3 / stride=1 / padding=1 / dilation=1 / groups=1。
最初の保護比で 6.8–10.3% 短縮し、保護前よりも全条件で短かった。
変更しない i8 の最初の保護と最終実装は同じ WGSL で、対照計測の差は −0.62〜+0.06%。

この表には executor が通常は 32 行タイルを選ぶ Cout も含む（1 / 5 / 32 / 65 / 129）。
これは 64 行変種の全域を検証するためであり、そうした製品形状にこの改善が適用されるという意味ではない。
モデル全体の速度改善率と M2 / ブラウザへの効果は未測定。

生成コードが変わった内部キーも更新した。Conv1d igemm は v4、Conv2d igemm は v3、i8 格納の linear GEMM は v3、W4A8 は v5。
未変更の i8a8 / 他の linear 格納 / 直接 conv のキーは維持する。履歴文書内の旧計測キーは書き換えない。

再開後の検証:

- codegen と geometry: 83 passed（`codegen-final.log`）。
- conv1d / conv2d の直接カーネルとのビット一致、i8a8、skinny GEMM: 26 passed（`gpu-parity-final.log`）。
- 最終差分の全体 `deno task verify`: **2,816 passed / 740 steps / 0 failed / 5 ignored、24m34s**（`verify-final.log`）。fmt・lint・型検査と、既存の GPU / asset 検査を含む全体が成功した。初回の静的検査で新規 Irodori テストの `require-await` 違反を検出し、Promise を直接返す形へ直して全体を再実行した（初回ログは `verify-static-first.log`）。

## 9 月 6 日および直近レビューの棚卸し

9 月 6 日のレビュー 1–4 は未処理の提案集としてそのまま実装できる状態ではなかった。
その後のコミット、現在の実装、backlog / perf-ledger に対応済み・棄却済み・継続中が混在していた。

| 項目                                                                            | 現時点の整理                                                                             |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| lm_head i8 GEMV                                                                 | K-16、`5ddd186` で実装済み                                                               |
| decode QK の並列縮約                                                            | K-14、`cce129d` / `4182b8b`、さらに `06f82df` で M≤8 に拡張済み                          |
| prefill QK / PV のタイル化                                                      | K-13、`ad8a4b9` / `39d5e4e` で実装済み                                                   |
| 小 M linear、shape 切替時の backing 保持                                        | K-21 `5701262`、H-15 `c7120f2` で実装済み                                                |
| PLE の cache hit 優先・重複排除・区間読出し、topK の bounded heap、会話 KV 継続 | 現行実装あり。今回 PLE の dispose 競合と履歴所有権を追加修正                             |
| MTP（公式 drafter による投機デコード）                                          | ADR 0096 の段階実装と自己採算ゲートまで到達。9 月 6 日の新規導入案はそのまま再実装しない |
| K-17 exp 共有、permute 消去、KV 量子化                                          | 既存の不採用理由・採算条件を尊重。今回再実装しない                                       |
| Anima DiT 内の反復常駐、topP のみの全語彙 sort、M2 小 M の機序                  | backlog の未完項目。既対応扱いにしない                                                   |
| 9/7 audit: vec4 共有タイルの競合                                                | `14f973f` で対応済み。今回の端チャネル読出しとは別原因                                   |
| 9/7 audit: layer_norm の低分散、W4A8 中間 overflow、長行 cumsum / 非最終軸 sum  | 数値仕様と新カーネルの設計を伴う。既存契約・backlog を維持し、今回許容誤差を変更しない   |
| 9/8 MTP レビュー: chunkLength 捕捉、pendingCommit の外部変更                    | `844b79b`、trace 上限は `bcd2539` で対応済み                                             |

9/3・9/4 のレビュー残件、9/6 performance-investigation、9/7 kernel-op-exploration も序盤の棚卸し対象とした。
これらの全候補を再ベンチしたという意味ではない。未検証の部分を既存結果から完了扱いにしていない。

## 継続候補と判断点

- **同一出力先への並行 publication（3/V04）**: `artifacts.staged_publication` は固定の .staging / .old を共有する。
  単一 writer の復旧テストは通るが、複数 writer の安全性は未検証。同一 final の同時書込みを許容するか、ロックの責務と回復方法を先に決める。
- **PLE leader の中断共有（3/V01）**: 現在の明示契約。リクエストごとに中断を分離する変更は、共有読出しの寿命設計と分けて扱う。
- **小ファイルの取得本数（3/V03）**: 現行の byte 予算だけでは本数が多くなりうる。ネットワーク実測なしに固定本数制限を追加しない。
- **Anima DiT の反復常駐（2/P02）**: まず stage 内の固定条件 upload と readback の費用を測る。次に stage 内だけで常駐化し、Euler / DPM++2M、イベント、abort、ピーク VRAM を検証する。
  stage 間や生成間で Session を保持する案は寿命とメモリの別設計を要する。
- **Irodori 条件 K/V の事前計算（2/P01）**: recipe 分割と対応するランタイム入力の設計が必要。固定条件部分の費用 → 一部層の比較 → 全層の順に進め、WAV と潜在の一致、追加常駐量を検収する。
- **Irodori の進捗通知と常駐経路（2/P03）**: `onEvent` または GPU timing が有効ならホスト経路になる現行分岐を確認した。イベントなし・空のイベント callback・実際の進捗表示・診断 callback のみ・GPU timing の 5 条件で、callback の費用と経路切替の費用を分けて測る。常駐経路への適用には完了通知と query 回収の設計が必要であり、今回の cleanup 修正には含めていない。
- **MTP の dense drafter head、stage 間同期、product PLE**: 既存の段 4 の研究に沿って run 内の時間を帰属させる。今回の observer 除外は計測の修正であり、これらの高速化実測ではない。
- **online attention / backend 別 geometry / 低精度演算（2/P04–P06）**: 長系列と形状別の作業量・メモリの基準値が先。既存の op tolerance と既定の u32 一致を維持できる範囲から段階化する。
- **GPU 検証の環境間差（2/V01）**: 今回 RTX の実行を行う。M2、ブラウザ、CI の必須 GPU ジョブ導入は未検証・未変更。

大きなモジュール書き直し・公開 API 変更・新規カーネルはこの修正へ混ぜていない。
推奨する次の性能調査は、既存 backlog と整合する Anima DiT stage 内の常駐化の費用計測。MTP を優先する場合は既存 research の run 内の host/readback 帰属を継続する。

## 検証記録

この節のログは `2026-09-10_review-and-fix/` 内。再開後の最終検証は前節の `2026-09-10_review-guard-cost/verify-final.log` を参照。GPU テストと GPU ベンチは並走させていない。

- 修正前 TypeScript: 153 tests / 78 steps 成功（`karume-review-baseline-ts.log`）。
- 修正前 exporter 関連: 130 tests 成功（`karume-review-baseline-python.log`）。レビュー環境の古い torch で報告された rank lowering の失敗は現行環境で再現しなかった。
- exporter 全体: 3,143 passed / 1 skipped（`karume-review-exporter-full.log`）。
- export-recipes 全体: 2,757 passed / 4 skipped（`karume-review-recipes-full.log`）。
- ストリーム関連: 最終 142 tests / 138 steps 成功（`lifecycle-final.log`）。終了値の Promise 拒否も含む。
- PLE と履歴所有権: 48 tests / 48 steps 成功（`karume-review-ownership.log`）、overflow の追加検査を含む最終履歴テストは `history-final.log`。
- codegen: 75 tests 成功（`karume-review-codegen.log`）。
- GPU 端数形状: 26 tests 成功（`karume-review-gpu-boundaries.log`）。conv1d / conv2d parity、i8a8、skinny GEMM。
- hub: 53 tests / 5 steps 成功（`karume-review-hub.log`）。
- Irodori 構築と共通 cleanup: 18 tests 成功（`karume-review-irodori.log`）。追加の実 GPU 多重故障テストも 1 test 成功（`irodori-cleanup-final.log`）。本物の解放後に故障を注入し、試験自体で資源を残さない。
- Python の変更 4 ファイルで Ruff lint / format check 成功。
- 再開前の全体 `deno task verify`: 2,814 passed / 740 steps / 0 failed / 5 ignored、24m31s（`verify.log`）。既存の GPU / asset 検査を除外せず実施。
- 全体検証の開始後に加えた終了ラッパーの強化と overflow 所有権検査は `lifecycle-final.log`、Irodori の新規ファイルは `irodori-cleanup-final.log` で別途検証した。最終の辞書構築への型注釈を含む公開面の型検査は `check-final.log`。
- 初回停止後、利用者の明示的な続行指示で再開した。追加の実測と最終検証は「再開後の調査と採用」を参照。この時点では未コミット。後続の利用者依頼で行った分割コミットは次節を参照。

## 分割コミットと再開用の引き継ぎ

利用者の追加依頼で、検証済み差分を作業範囲ごとに 10 個のコード修正と 1 個の文書コミットへ分けた。
同じファイルに混在するストリーム終了・履歴所有権・採算計測・資産表構築はハンク単位で分離した。
作業ツリーのコードは変更せず、各コミット前にブランチ・ステージ差分・保存した SHA-256 を照合した。

| コミット  | 作業範囲                                     |
| --------- | -------------------------------------------- |
| `d4b351c` | 動的 view の定数化を単位軸の変更に制限する   |
| `555f358` | 記号依存マスクの safe_softmax を維持する     |
| `03e96c1` | 生成ストリームの終了通知と解放を整合させる   |
| `edf76f4` | 会話履歴のメッセージ所有権を分離する         |
| `3bdaea6` | PLE 破棄後のキャッシュ再登録を防ぐ           |
| `2086d75` | Irodori の多重障害でも全資源を解放する       |
| `2f85b2a` | 投機の採算計測から診断 callback の費用を除く |
| `0d9284e` | 空 draft の抽選で履歴コピーを省く            |
| `b054dad` | 資産辞書の構築で二乗コピーをなくす           |
| `732734f` | GEMM の端チャネル読みを範囲内に収める        |

最後の文書コミットは本節・backlog・perf-ledger・ACTIVE_DESIGN の索引を含む。
最終一覧と内容照合用の SHA-256 は [分割コミット記録](../../outputs/bench/karume/2026-09-10_review-and-fix/split-commits/) に保存する。

### 検証の扱いと今後の作業方針

- **今回限りの明示承認**: 利用者は「今回は検証結果を再利用する」と回答した。
  直前の `verify-final.log`（2,816 passed / 740 steps / 0 failed / 5 ignored）と Python 両パッケージの成功を再利用した。
  各中間コミットで全体検証を再実行したという意味ではない。分割の最終内容と検証済みコードの一致を確認し、追加文書は整形・リンク・数値を検査する。
- **今後の恒常的な希望**: 計測・原因の帰属・小さな修正・検証を一つの作業単位として完了させ、各単位を順次コミットする。
  次の独立した作業へ差分を溜め込まない。通常の各コミット前の `deno task verify` と、exporter / recipe 変更時の両 Python 検証は継続する。
- 変更は現在の `codex/review-and-fix` のみ。他ブランチ、特に main は変更しない。push は利用者が行う。
  既存のローカル `AGENTS.md` は利用者の未追跡ファイルとして変更せず、コミットにも含めていない。
- サブエージェントによる序盤の棚卸しは終了。続きの調査・修正も主担当で進める。
  想定外の問題は停止して指示を仰ぐ。範囲外アクセス保護に伴う費用の調査は明示承認済み。

### 次のコンテキストで最初に読むもの

1. `git status --short` / `git branch --show-current` / 直近の `git log` で引き継ぎ後の差分と作業場所を確かめる。
2. `CLAUDE.md`・`.claude/ACTIVE_DESIGN.md`、本記録、backlog の now を読む。
   本記録の開始点 `4dca96a` を未修正の HEAD と誤認しない。修正済みのレビュー項目を再実装しない。
3. 生データは `outputs/bench/karume/2026-09-10_review-and-fix/` と `2026-09-10_review-guard-cost/`。
   ベンチの時刻は同一走行内の比較だけに使い、異なる走行・別 GPU の絶対値を混ぜない。
   f32 Conv1d の採用範囲は vec4・64 行タイルのみ。Cout=1 / 5 / 32 / 65 / 129 は通常 planner が 32 行を選ぶため、64 行の単体改善を製品性能と扱わない。
4. 次の推奨調査は **Anima DiT の stage 内における固定入力 upload・latent readback の費用計測**。
   現行 Session / stage の寿命と関連 ADR を読み、既存の実行経路で時間を帰属するところから始める。
   常駐化の実装前に、影響範囲・Euler / DPM++2M の数値一致・イベント・abort・ピーク VRAM の検証計画を具体化する。
   MTP を優先する場合は ADR 0096 と `2026-09-09-mtp-stage4.md` の最新節を読み、run 内の host / readback の帰属を続ける。
5. M2 / ブラウザの追試、同一出力先への複数 writer、Irodori 進捗通知時の常駐化などは上記「継続候補と判断点」の未完項目。
   大きなモジュール書き直し・公開 API 変更・新規カーネルの実装は、設計と検証計画を示して判断を求める。

分割コミット後に実行中の検証・ベンチはない。新しい出力先を作り、GPU テストとベンチを並走させない。
この環境では sandbox の namespace 作成が失敗するため実行に昇格を使用したが、再開時に環境が変わっていれば現状を確認する。
