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
  最初の 11 件の分割時点では `AGENTS.md` を未追跡のまま保持した。後続の明示依頼で、現在のブランチと順次コミット方針へ更新して追跡対象に加えた。適宜編集する許可も得ている。
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

## モデル横断の追加調査（2026-09-10）

利用者の追加依頼により、`dad9078` の後からモデル別・共通の最適化と追加モデルの事前調査を開始した。
序盤の Sol 3 エージェントは性能候補、追加 LLM、動画モデルを読み取り調査し、その後の実測・変更は主担当が行う。
生データは別ディレクトリ [2026-09-10_model-optimization](../../outputs/bench/karume/2026-09-10_model-optimization/) に保存する。
以下の数値は RTX 3080 Ti / Deno 2.9.6 / Linux での観測であり、M2 / ブラウザ実測ではない。

### 共通サンプラーの top-p 単独指定

**採用**: `samplerDistribution` が `topK` 無し・`topP < 1` のときに使う全語彙の比較ソートを、
f32 の順序キーによる 4 pass の安定 radix sort にした。全語彙を走査する O(V) の実装で、
追加の作業配列は Int32 2 本・Uint32 1 本と 1,024 個のカウンタ（V=262,144 で約 3 MiB）。
配列の最大長は従来の token id と同じ i32 の範囲に従う。共有キャッシュ・依存追加・公開 API 変更はない。

- 既存の数値降順・同値 id 昇順を維持し、+0 / -0 のキーを共通化する。NaN / +Inf の拒否条件は維持。
- f64 softmax の加算順も同じなので、累積確率の直前・一致・直後で残る候補と確率がビット同一になる。
- top-k 有りと greedy は従来経路のまま。Gemma 配布既定の top-k 64 / top-p 0.95 は今回の対象外。

合成ベンチは V=32〜262,144 の 7 サイズ × 4 分布、各 2 暖機 + 9 回、前後の順序を反転した 504 標本。
1 標本は小語彙 20 回、大語彙 3 回の平均。28 条件で候補 id と確率 f64 の全ビット一致を確認した。
V=262,144 の中央値は次のとおり。

| 分布              | 比較ソート ms | radix ms | 倍率 |
| ----------------- | ------------: | -------: | ---: |
| 一様乱数の logits |        90.083 |   12.143 | 7.42 |
| 昇順              |        16.789 |    9.399 | 1.79 |
| 13 種の同値       |        37.689 |    8.297 | 4.54 |
| 末尾 64 語へ集中  |        15.561 |    7.793 | 2.00 |

さらに既存 `mtp-bench` の dialogue / greedy / 32 token から Gemma E2B の logits 24 行を採取し、
温度 0.7 / 1.0、top-p 0.95 で前後を再生した。各条件 1 暖機 + 5 回、順序反転、全 48 条件で
候補と確率がビット同一。各温度 240 標本の中央値は **88.102 → 11.542 ms（7.63 倍）**、
**87.688 → 11.188 ms（7.84 倍）**。CPU の抽選だけの測定で、モデル全体や top-k 既定生成の倍率ではない。
小語彙では約 1 µs の逆行例があるが、大語彙向けの恣意的な切り替え閾値は追加しない。

再現資料: `sampler-spike.py` / `sampler-before.ts` / `sampler-bench.ts`、`sampler-radix.jsonl` と
`sampler-radix-summary.json`、`capture-logits.ts` / `gemma-logits.f32` / `gemma-logits.json`、
`sampler-replay.ts` / `sampler-replay.jsonl` / `sampler-replay-summary.json`。
本体テストは符号付きゼロ・subnormal・同値・view の範囲・累積境界前後・大語彙の整列で比較ソート参照と照合する。
検証: 対象 33 tests / 3 steps、全体 `deno task verify` は **2,817 passed / 743 steps / 0 failed / 5 ignored（25m17s）**。
ログは `sampler-tests.log` と `verify-sampler.log` に保存した。

### greedy の NaN 検査と最大値検索

**採用**: 温度 0 の `samplerDistribution` は、NaN の全語彙検査と argmax の探索を 1 回の走査にまとめた。
NaN は最大値の位置以外も必ず拒否し、最初の NaN の位置を従来と同じ文言で報告する。
+Infinity / 全 -Infinity の拒否、同値の先勝ち、view の範囲、乱数の消費数を維持する。
温度が正の確率生成経路は従来の事前検査を使う。追加のキャッシュや作業配列はない。

発端は Gemma freeform / greedy / 128 token / 1 round の V8 CPU profile。
42.28 秒の取得区間では、自己時間の標本が argmax に約 450 ms、NaN 検査に約 110 ms あった。
区間にはロードと暖機も含む。native `mapAsync` 約 15.55 秒、idle 約 19.81 秒という標本は
CPU 使用率ではなく、GPU 待ちを含む呼び出しへの時間帰属である。JS 全体を書き換える根拠にはしない。
`profile-host.ts` / `profile-target.ts`、`gemma-host.cpuprofile.json`、`gemma-cpu-profile-summary.json` に保存した。

同じ実 logits 24 行を 2 暖機 + 9 回、前後の順序を反転して再生した。
1 標本は 20 回の平均、各実装 216 標本。token は全件同じで、中央値は
**0.557 → 0.194 ms（2.87 倍、約 0.36 ms 短縮）**。
これは CPU の greedy 選択だけの数値で、モデル全体の倍率ではない。
`prepare-argmax-spike.py` / `argmax-replay.ts`、`argmax-replay.jsonl` / `argmax-replay.log` が再現資料。
境界テストは先頭・末尾の NaN、NaN と +Infinity の共存、範囲外に NaN を持つ view、±0 の同値を追加した。
対象検証は **34 passed / 3 steps / 0 failed**（`argmax-tests.log`）。
全体 `deno task verify` は **2,818 passed / 743 steps / 0 failed / 5 ignored（24m44s）**
（`verify-argmax.log`）。

### 追加 LLM の実行と量子化別比較

**実験用の系列を作成し、既存 runtime で検証した**。公開モデルの登録・配布 manifest・製品 pipeline の追加はまだ行っていない。
対象は Gemma 4 E4B、MiniCPM5-2B、Qwen3-0.6B。公式重みは `inputs/<family>/`、変換結果は
`outputs/series/*-2026-09-10-probe/` に分離した。取得 revision・ファイル長・モデルカードのライセンスは
`model-census.json` に保存。既存の配布資産を上書きせず、外部実装の複製やリポジトリの clone は行っていない。

MiniCPM5-2B と Qwen3-0.6B は、既存 MiniCPM recipe の chunk wrapper と state 変換を使う実験台本
`export-llm-probe.py` / `export-llm-quant-probe.py` で変換した。モデル本体は公式 Transformers の
`AutoModelForCausalLM` を読み、RoPE の表引きは置換前の全位置照合を維持する。
Qwen は 28 attention / 56 KV slots、MiniCPM は 42 / 84。RoPE 表 256、context 128、prefill chunk 16 の短文試験である。
f16 は格納だけを圧縮し演算は f32。i8 は既存 per-channel、i4 は decoder linear を group 32 とし、
embedding / lm_head は共有関係を保った i8 にする。f16 試作時に未丸めの RoPE 定数表の格納を拒否されたため、
表を f32 と明示した。失敗した試作のログ・部分出力を別名で保存し、格納検査の条件は変更していない。

`run-llm-probe.ts` は各量子化と同じ重みの CPU 参照に対し、3 入力の prefill 全 logits を
atol=1e-3 / rtol=0 で検査し、8 token の greedy 継続を厳密照合した。7 形式すべて成功。
続く `llm-quant-bench.ts` は 1 device / 1 process 内で各形式を往復順にロードし、各 3 入力 × 64 token を生成した。
先頭の decode 7 回を暖機として除いた 56 回 × 6 ケース = 各形式 336 標本の中央値を示す。
この性能試験も全 logits と先頭 8 token を照合する。残る 56 token の CPU 一致は主張しない。
EOS 後も固定長で計算する台本なので、chat 応答時間・品質の評価とは区別する。

| モデル      | 格納         | 変換済み shard の総 byte | 定常 decode ms/token | prefill 最大絶対誤差 |
| ----------- | ------------ | -----------------------: | -------------------: | -------------------: |
| Qwen3-0.6B  | f32          |            2,385,411,200 |               42.187 |        0.000240 未満 |
| Qwen3-0.6B  | f16          |            1,193,441,600 |               42.700 |        0.000246 未満 |
| Qwen3-0.6B  | i8           |              599,478,792 |               21.782 |        0.000144 未満 |
| Qwen3-0.6B  | i4 / head i8 |              432,956,648 |               21.096 |        0.000084 未満 |
| MiniCPM5-2B | f32          |           10,068,270,544 |               87.938 |        0.000202 未満 |
| MiniCPM5-2B | i8           |            2,522,805,240 |               28.820 |        0.000100 未満 |
| MiniCPM5-2B | i4 / head i8 |            1,776,451,608 |               26.596 |        0.000294 未満 |

数値の正本は `qwen3-06b-quant-bench.jsonl` / `minicpm5-2b-quant-bench.jsonl` と各 `-summary.json`。
初回の短い試験値は `llm-quant-summary.json` に別途残した。別プロセスだった初期値を、この表の倍率計算へ混ぜない。
Qwen の f16 は容量を半減しても速くならない。現行 linear は i8 / i4 の小 M を GEMV へ送り、f16 は
M=1 でも共有タイルの GEMM を使うので、**f16 M=1 の計算方式が次の候補**となる。
これは原因候補であり、専用 kernel による改善を測定済みという意味ではない。
`f16-gemv-plan.md` に影響範囲・逐次 K 順序・u32 一致・単体 1.3 倍または decode 10% の門を記し、試作の判断を求めた。

Qwen はさらに公式 chat template の `enable_thinking=False` で France / Japan / WebGPU の 3 問を確認した。
`chat-refs.py` / `run-chat-probe.ts`、`chat-fixtures/`、`qwen3-06b-{f16,i8,i4}-chat-gpu.jsonl` が根拠。
9 条件すべて、prefill 全 logits（最大絶対誤差 0.000181 未満）と EOS または 32 token までの CPU 継続が一致した。
ただし **一致は品質を保証しない**。f16 / i8 でも日本の首都を大阪と答え、i4 は France を Lyon、日本を京都と答えた。
これらは export 用ラッパーの CPU 参照にも現れた応答であり、GPU 固有の誤差とは断定しない。
後述の独立検証で見つかった差は検証台本の RoPE 丸めに帰属し、条件を合わせた公式 CPU でも f16 の 3 問は全 token 一致した。
単純 i4 を品質検収なしに採用しない。
公式 Qwen は thinking mode で greedy を推奨していない。本試験の greedy は数値検収用である。
[Qwen3-0.6B の公式モデルカード](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/README.md)。
MiniCPM の公式 chat 用 `chat_template.jinja` はこの時点では未取得で、製品 chat の成立までは検証していない。

E4B は `export-e4b-probe.py` で decoder を main linear i4 / embedding i8 とし、3,159,221,160 byte の系列を作った。
42 attention / 48 slots、sliding / full attention の共有元は層 22 / 23。CPU 参照用の PLE は必要な 38 行だけを読み、
全語彙の PLE sidecar・manifest・drafter はまだ作っていない。モデルを meta 上に構築して重みを assign し、
共有 embedding の同一オブジェクト性を `tie_weights()` で復元した。検証用 export の最大 RSS は約 25.6 GB。
`run-e4b-probe.ts` は 3 入力 × 8 step の logits / hidden を atol=1e-2 で照合し、argmax id は全件一致。
最大絶対誤差は logits 0.000191 未満 / hidden 0.000271 未満だった。
短い decode の中央値は入力別 37.26 / 36.68 / 35.52 ms。これは動作確認の付随値で、形式間の速度比較ではない。
`export-e4b.log` / `e4b-gpu.jsonl` / `e4b-gpu-summary.json` に保存した。

追加のメモリ試算は `model-memory.ts` / `model-memory.json` に保存した。
既存 `PreparedModel.estimate`、chunk 64、capacity 4,096、binding 上限 256 MiB、backing 予算 256 MiB の条件で、
Qwen3 i4 は weight 412.56 MiB / KV 896.00 MiB / 勘定済みピーク 1,564.56 MiB、
MiniCPM5-2B i4 は 1,693.69 / 336.00 / 2,285.69 MiB、E4B は 3,012.23 / 168.63 / 3,436.85 MiB となる。
小型 Qwen でも KV head が 8 本あるので、長い context の KV は MiniCPM より大きい。
これは **構造からの試算**。E4B の host PLE、upload staging、退役待ち、その他 `unaccounted` はピーク値に含まれず、
ブラウザでの動作可否の保証ではない。今回の Qwen / MiniCPM 資産の RoPE 表は 256 位置なので、
長文を実行するには位置表の拡張または host RoPE の設計と追加検証が必要である。

### Gemma E2B mobile QAT は同じ構造の別量子化変種

[公式モデルカード](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/blob/main/README.md) と
[config](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/blob/main/config.json)、
[Transformers の gemma_quant](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/gemma_quant.py)
を確認した。35 層・hidden 1536・FFN 6144・query / KV head 8 / 1・PLE 256 などの骨格は E2B と同じ。
一方、量子化を考慮した学習による重みと演算を持ち、現行 E2B の PTQ i4g32 と同一の数値モデルではない。

主 embedding / lm_head は INT2、PLE embedding は INT4、MLP は前半 15 層 INT4 / 後半 INT2、
attention INT4、per-layer gate / projection は INT8。出力行ごとの scale と整数 packing を使い、group 32 ではない。
さらに SRQ（静的な再量子化）として活性へ `x / scale → round → clamp[-128,127] → ×scale` を適用する。
scale=0 の場合の処理も公式実装に従う必要がある。重みを展開して現行 i4 へ再量子化するだけでは、この意味を再現できない。
現 runtime には INT2 格納と `round` op がなく、`docs/op-vocabulary.md` の round の候補記述を実装済みと誤認しない。

今回取得したのは config・tokenizer・カードなどのメタデータだけで、QAT 重みの export / GPU 実行は未実施。
追加するなら、まず SRQ と整数 packing の数値契約・tied weight の扱いを決め、CPU 参照と 1 linear / embedding の
検収から進める。通常 E2B の配布設定へ黙って置き換えない。

### 動画生成の事前調査: Wan と MiniMax H3

以下は公式構成と現行 IR からの調査・試算であり、動画モデルの GPU 実測ではない。
初期調査 `initial-video-feasibility.md` の結論を主担当が公式カード・実装・公開ファイル一覧で確認した。
`model-census.json` に取得時の revision とファイル容量を保存した。動画の重みは取得しておらず、外部ソースの複製・clone もしていない。

最初の候補は **Wan2.1-T2V-1.3B の小さな DiT 単体試験**とする。
[公式実装](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py) は
30 block / hidden 1536 / 12 head / FFN 8960、時間・空間の RoPE を使う。
[VAE](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py) は時間方向を含む causal Conv3d で、
時間 4 倍・空間 8 倍圧縮、DiT の patch は時間 1 × 空間 2 × 2。
公式の 832×480・81 frame は `21×30×52 = 32,760 token` になる。

| Wan2.1 の中間値                       | 単純に実体化した容量の試算 |
| ------------------------------------- | -------------------------: |
| hidden `[32760,1536]` f32             |                 191.95 MiB |
| FFN `[32760,8960]` f32                |                  1.093 GiB |
| 全 self-attention score・12 head・f32 |                  47.98 GiB |
| 同 score・f16                         |                  23.99 GiB |

Anima の段ごとに Session を開閉する寿命管理、linear / norm / attention は再利用候補である。
一方、現 IR は rank 1〜4、VAE は rank 5 と Conv3d が必要で、画像用 patchify / spatial tiling をそのまま適用できない。
既存 attention の row block は一時容量を下げるが、二乗の計算量は残る。
動画では長系列の online attention、FFN の分割、時間方向の VAE cache を別々に設計・検収する必要がある。
self-attention の K/V は denoise ごとに変わるので、LLM の KV cache を流用して step 間に残すことはできない。

順序は、①小さな固定 text 条件で DiT の CPU/GPU 照合、②実 token 長の attention / FFN 容量・時間、
③causal VAE の短い時間 chunk、④公式 scheduler と各段を結ぶ経路、とする。
[公式カード](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/README.md) の VRAM 8.19 GB は
PyTorch の offload / T5 CPU を伴う条件なので、ブラウザの必要メモリとして転用しない。
[Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2) は VAE の空間圧縮が 16 倍となり、
704×1280・121 frame は 27,280 token になるが、重み・公式実行条件が大きく、動画基盤を作った後の候補とする。
Wan の公式カードは Apache-2.0。派生配布を作る段階では同梱する text encoder / VAE の条件も確認する。

**MiniMax H3 は公開重みを持つ別の動画・音声生成モデルで、Hailuo-02 / 2.3 の別名ではない。**
[公式説明](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/README.md) とファイル一覧では、
33B Transformer の BF16 重みだけで **66,280,430,144 byte**、ほかに Qwen3-VL-32B text encoder 約 66.7 GB と
visual VAE 約 10.4 GB がある。約 13B の AdaLN branch は事前計算で省略可能とされるが、
従来の索引の「42.5 GB」を公式配布物のサイズとする根拠にはならない。
初回公開は full attention の経路で、sparse attention は未公開。公開 Base 768p と、未公開の後段を含む 2K 製品経路も区別する。
したがって H3 は構造の調査対象に留め、今回のブラウザ実装には着手しない。

[H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) は Apache-2.0 ではない。
適用地域から EU・英国・韓国・米国を除き、年間売上 2,000 万 USD 超の商用利用には事前の書面承認を求める。
商用 UI の表示・再配布・利用制約の条件もある。実装・配布を検討する際は、その時点の想定地域と形態に照らして確認する。

### ホスト待ちと既存融合の追加測定

実行台本は `gemma-phases.ts`、結果は `gemma-phases-extract.jsonl` / `gemma-phases-extract-summary.json` へ保存した。
extract・prompt 4,845 / capacity 8,192・greedy・64 token の診断走行で、定常 plain は
run 壁 28.482 ms / encode 3.247 ms / map 24.938 ms、そのうち map の同期区間は 13.855 ms。
draft は壁 14.663 ms / encode 1.107 ms / map 13.487 ms、map 同期区間 2.403 ms だった。
error scope の pop は約 0.005 ms に留まり、pop と map の並列化を速度策とする根拠はない。

非同期 map の約 11 ms の床は、実行版と同じ
[Deno v2.9.6 の `buffer.rs`](https://github.com/denoland/deno/blob/v2.9.6/ext/webgpu/buffer.rs#L239-L261)
の待ち方と整合する。device poll の後に 10 ms sleep する future と通知受信 future を `try_join!` で待つため、
通知だけが先に来ても sleep 側の終了を待つ。このソース確認は実測への帰属であり、Deno 自体を変更したものではない。
純 TypeScript / Web 標準の runtime でこの内部待ちを解除する口は見つかっていない。
`gemma-host.cpuprofile.json` の native `mapAsync` に約 15.55 秒が載るが、待ちを含む標本なので CPU 演算時間と解釈しない。

既存融合 attention の分割数は `attention-split.ts` で `[1,16,4096,128]` を 1 / 2 / 4 / 8 / 16 / 32 / 64、
往復順に測った（a8 / score f16）。14 試行すべて u32 一致・timestamp clamp 0。
1 分割は一時容量 601.01 MiB、GPU 6.46 / 6.98 ms、2 分割は 344.76 MiB、6.77 / 6.80 ms。
4 分割は 216.63 MiB、6.99 / 9.69 ms、8 分割は 152.57 MiB、8.42 / 10.90 ms となり、
細分割では dispatch 数が増え、遅くなった。**分割増を高速化としては採用しない**。
`attention-split.jsonl` が数値の正本。これは attention 単体の容量であり、Anima 全体のピークではない。

Anima の固定入力は `anima-inputs.ts`、1024×1024 / CFG=1 / Euler 8 step / seed 42 で、
host → RoPE 常駐 → RoPE + text 条件常駐 → 逆順の 6 回を比較した。
1 forward の upload は 7,413,760 → 3,219,456 → 1,122,304 byte へ減り、全 step / PNG の SHA は一致したが、
GPU buffer の観測ピークは全試行 2,775,563,888 byte のまま、全体壁は 11.59〜12.06 秒の変動範囲だった。
転送の host 呼び出しは約 0.5 ms、DiT は約 800 ms/forward なので、この条件で有意な全体利得は示せない。
**固定入力の常駐化だけでは採用しない**。CFG>1、latent / scheduler の GPU 常駐化は別の未検証範囲。
数値・ハッシュは `anima-inputs-quick.jsonl` に保存した。

`abort-timer.ts` では既存 `settleAbort` の CPU 費用を測り、signal なし約 0.0003 ms、
未 abort の signal あり約 2.07 ms/境界だった（Deno、5×256 回）。
過去の MessageChannel 案は timer 由来の中断を最初の境界で拾う契約を破るので再採用しない。
中断契約を変えずに短縮できる手段は未確認。ブラウザでの値も未測定である。

### 独立した公式 CPU 検証で停止（22:09 UTC 頃）

`qwen-official-chat.py` は Qwen3-0.6B を公式 `attn_implementation="sdpa"` のまま読み、
既存参照と同じ f16 丸めを重みに施し、公式 non-thinking chat template の入力 id を厳密照合した。
RoPE 表への置換・独自 attention 登録・export 用ラッパーは使わない。
France 質問の prefill 全 logits を既存のラッパー CPU 参照と比較したところ、
最大絶対差 **0.003142833709716797**、atol=1e-3 / rtol=0 を超える要素は **70,138 / 3,950,336** だった。
`qwen-official-chat.log` に失敗を保存した。生成継続の比較に到達する前に停止している。

これは GPU 対ラッパー CPU の既存検証とは別の門である。
attention の演算経路による丸め差なのか、ラッパーの意味差なのか、生成 token に影響するのかは未判定。
許容差を変更せず、利用者の「想定外の問題は一旦止める」指示に従って追加実験・修正を停止した。
公式モデルへの忠実度を確認するまで、Qwen の chat 品質に関する帰属と新形式の製品採用を保留する。
再開時は同じ重み・入力の同一性、RoPE、mask、公式 SDPA と登録 attention の層別差を順に切り分ける。

直前の W1 比較は cold 3 課題 / warm 2 課題 × base/skip の **10 条件**が成功し、全 token 列一致。
`w1-ab-summary.json` に比率を保存した。初回 plain 観測の除外に安定した速度利得はなく、製品へ未採用。
台本に含めた warm extract は既存 bench が非対応として拒否したため、成功件数から除外した。
`w1-extract-warm-skip.log` と `w1-ab-progress.log` がこの台本条件の誤りを記録している。

後続の `radix-width.ts`、`extract-drafter-real-bench.py` / `drafter-real-gemv.ts`、
`anima-w4-geometry.ts` は `run-next-trials.py` の最初の検証失敗で **未実行**。
GPTQ の校正比較も台本準備のみ。新しい f16 GEMV kernel は試作の判断待ちで、実装していない。
Chrome は隔離した一時環境で NVIDIA ANGLE Vulkan を認識したが WebGPU adapter が得られず、
ブラウザのモデル実行・性能検証には到達していない（`browser-diagnose-*.json` / `browser-info.log`）。

### Qwen の不一致を検証台本の RoPE 丸めへ帰属（2026-09-11）

利用者の継続承認後、`qwen-attention-attribution-fixed-rope.py` で同じモデル実体の
重みを固定し、公式 / 登録 attention、暗黙 / 明示 mask、通常 / math SDPA、RoPE 表の有無を切り分けた。
原因は **独立検証の台本だけが `round_weights_to_f16(model)` を RoPE 置換前に呼んでいたこと**。
この既存関数は仕様どおりパラメータと f32 バッファを丸める。公式 RoPE の
`inv_freq` / `original_inv_freq` は各 64 要素中 63 要素が変わり、最大変化は 0.00017815828323364258 だった。
一方、export 用参照では丸め前に RoPE を f32 の位置表へ置換するため、解析的な周波数は丸めない。
前回の独立検証はこの条件が揃っていなかった。runtime / exporter の数値不具合ではない。

パラメータのアドレス・更新世代が不変であることを確認したまま、周波数バッファだけを元へ戻すと、
France 質問の全 logits の最大差は **0.003142833709716797 → 0** になった。
通常 SDPA では mask の明示・登録 attention・RoPE 表のどの組み合わせでも既存参照とビット同一。
math SDPA を強制した場合だけ最大 0.00008392333984375 の演算経路差が残り、従来の atol=1e-3 / rtol=0 を満たした。

さらに、元の周波数を保った公式 CPU 実装で 3 問を通常 / math SDPA の両方で再検証した。
通常 SDPA は **prefill 全 logits が 3 問ともビット同一**、math の最大差は 0.000084 未満。
継続は France 2 / Japan 10 / WebGPU 32 token、計 6 条件すべて既存参照と厳密一致した。
Japan の「大阪」という応答も再現し、GPU / ラッパー固有の異常ではないことを確認した。
この短い試験をモデル全般の品質評価へ外挿しない。

数値の正本は `qwen-attention-attribution-fixed-rope.json` / 同 `.log`。
最初の失敗と、周波数丸めを残した対照群も保存した。検証の atol・rtol・期待 token は変更していない。
本体コードへの修正は不要で、後続の既存カーネル設定比較へ再開できる状態となった。

### top-k の候補走査へ NaN 検査を統合

`topk-scan-replay.ts` は保存済み Gemma 4 の実 logits 24 行（語彙 262,144）に対し、
温度 0.7 / 1、topK=1 / 64 / 1024、topP=0.95 を測った。各条件 2 warmup + 7 round、
1 標本あたり 10 回、旧 / 新を往復順に実行し、各条件 168 標本を保存した。
変更は NaN 検査を top-k heap の充填・残りの走査へまとめるもの。token id 順に末尾まで検査するため、
候補から落ちる NaN、先行する Infinity、入力 view の範囲、最初の NaN を報告する契約を維持する。

| 温度 | topK | 変更前 CPU 中央値 ms | 変更後 ms |  倍率 |
| ---- | ---: | -------------------: | --------: | ----: |
| 0.7  |    1 |             0.977333 |  0.770966 | 1.268 |
| 0.7  |   64 |             0.951895 |  0.812093 | 1.172 |
| 0.7  | 1024 |             1.321099 |  1.216120 | 1.086 |
| 1    |    1 |             0.488674 |  0.386639 | 1.264 |
| 1    |   64 |             0.506086 |  0.403607 | 1.254 |
| 1    | 1024 |             0.965304 |  0.860144 | 1.122 |

全比較で候補 id と Float64 確率のビット列が一致し、NaN / Infinity / 同値 / 空入力の例外も一致した。
製品側は heap の検査統合だけを採用し、NaN が候補集合の外にある場合と加工後に発生する場合のテストを追加した。
数値の正本は `topk-scan-replay.jsonl`、集計は `topk-scan-summary.json`。
これは CPU のサンプラー単体の改善であり、生成全体がこの倍率で速くなる主張ではない。
greedy や topK 未指定時の処理は今回の変更に含まない。
