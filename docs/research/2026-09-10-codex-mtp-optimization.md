# レビュー指摘の再検証と修正（2026-09-10）

この文書は 2026-09-10〜11 時点の調査・実測スナップショットであり、恒久的な作業状態は [backlog](../backlog.md) を正本とする。

## 対象と環境

利用者が指定した `.claude/reviews/2026-09-11_chatgpt-reviews/` の 3 レビューを主担当が直接確認した。
ディレクトリ名の日付と実行日付は区別する。序盤のみ GPT-5.6 Sol / Terra に 9 月 6 日および直近レビューの棚卸しを委譲し、以後の調査・修正・検証は主担当が行った。
開始点は `4dca96a`、作業ブランチは `codex/review-and-fix`。最新の利用者指示を優先し、旧作業指示の `codex/mtp-optimization` へは切り替えていない。他ブランチへの変更はない。

実測は Linux / Deno 2.9.6 / RTX 3080 Ti（driver 610.57.04）で実施。
Python は `tools/exporter` / `tools/export-recipes` の `uv run --locked` 環境（Python 3.14.6 / PyTorch 2.13.0+cpu）。
レビュー修正時点では Apple M2 とブラウザは未実測。9/11 の Chrome 追試は後段に記録し、M2 はこの環境では未実測のまま。
外部リポジトリのクローンや外部リポジトリ由来の実装の複製は行っていない。

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
  停止条件はデータ損失・誤操作などのインシデント。性能低下・数値誤差は検証条件を保って調査を続ける
  （後述の「停止条件の解釈訂正」）。範囲外アクセス保護に伴う費用の調査は明示承認済み。

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

この節の調査時点で取得したのは config・tokenizer・カードなどのメタデータだけ。
後段で重みの部分取得と CPU 照合を追加したが、QAT 全体の export / GPU 実行は未実施。
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

### top-k の根の値を差し替え時だけ読み直す

NaN 検査統合後を基準に、heap の根の logit を候補が入れ替わる時だけ読み直す候補を比較した。
定常相で heap を変える箇所は `siftDown(0)` の直後だけなので、そこで局所変数 `cutoff` を更新する。
不採用になる各 token で `heap[0]` → `logits[...]` の間接参照を繰り返さずに済む。
NaN の検査順、同値の id 順、softmax の加算順は変えない。

`sampler-next.ts` は実 logits 24 行と昇順 / 降順 / 同値の 3 入力、各 262,144 語を使用した。
温度 0.7 / 1 × topK=1 / 64 / 1024、topP=0.95、2 warmup + 7 round、1 標本 10 回の往復順測定。
表は実 logits の各条件 168 標本の中央値である。

| 温度 | topK | 基準 CPU ms | 根の値を保持 ms |  倍率 |
| ---- | ---: | ----------: | --------------: | ----: |
| 0.7  |    1 |    0.771255 |        0.593991 | 1.298 |
| 0.7  |   64 |    0.847418 |        0.612445 | 1.384 |
| 0.7  | 1024 |    1.214434 |        1.066116 | 1.139 |
| 1    |    1 |    0.475284 |        0.238461 | 1.993 |
| 1    |   64 |    0.490574 |        0.256948 | 1.909 |
| 1    | 1024 |    0.856662 |        0.762831 | 1.123 |

全候補の id / Float64 確率のビット列、NaN / Infinity / 空入力の例外が基準と一致した。
候補が毎回入れ替わる昇順でも倍率 0.999〜1.273 で、計算量は元のまま。
温度だけの加工時に初期コピーを省く候補も併測したが、単独の改善は温度 0.7 で約 3〜5% に留まった。
今回は根の読み直しだけを採用し、温度用の追加分岐は入れない。
数値の正本は `sampler-next.jsonl`、実入力の集計は `sampler-next-summary.json`。
サンプラー単体の CPU 計測であり、生成全体の倍率ではない。

### 既存カーネルの実行形状を再比較（不採用）

`drafter-real-gemv.ts` は、配布済み Gemma drafter の i8 head の実重みを読み、
K=256 / N=262,144、同じ入力の独立した出力 128 本、heater 付き 5 round で比較した。
7 条件すべて出力 u32 一致・timestamp clamp 0。基準 c32u4 は 87.961 / 87.893 / 160.518 µs、
c32u8 は 89.322 µs、c64u4 は 173.814 µs、c16u4 は 160.938 µs、c64u8 は 120.987 µs。
**同じ基準のドリフトが大きく、安定した勝者は示せないので変更しない**。
反復数は 128 に固定しており、最速時の計測パスは約 11 ms で既存の目標 80 ms に届かない。
この制約と、単体の高速な値だけをモデル全体へ外挿できない点を残す。
正本は `drafter-real-gemv.jsonl`。抽出元を含む台本は `extract-drafter-real-bench.py`。

`f16-gemm-geometry.ts` は新カーネルを作らず、既存 GEMM の M=1 だけを 7 形状 + 基準再測定で比較した。
Qwen3 の N/K=2048/1024、3072/1024、1024/3072、計 24 条件は出力 u32 一致・clamp 0。
基準 → wgX=2 の候補 → 最後の基準は、それぞれ 79.329 → 78.579 → 90.198 µs、
95.064 → 93.839 → 95.068 µs、281.621 → 257.646 → 264.999 µs。
行方向のタイルをさらに小さくした候補や register 数を変えた候補は遅く、
**f16 が i8 より遅い問題を解消するような利得は出なかった**。既定は変更しない。
正本は `f16-gemm-geometry.jsonl`。新しい f16 GEMV の試作提案とは別の試験である。

Anima w4a8 は M=4096、N/K=2048/2048・8192/2048・2048/8192 を各 11 条件で測った。
`anima-w4-geometry.jsonl` の 33 条件はすべて出力 u32 一致・clamp 0。
出現数 168 / 28 / 28 による加重合計は基準 707.129 / 710.973 ms、
最良の regM=8 / regN=4 / wgX=8 / wgY=16 / tileK=16 が 665.219 ms だった。

この候補を **実モデルの最後の検収で棄却した**。
既存 `anima-v1.0-i4-dyn` transformer と現行 base の text encoder / VAE / RoPE を使い、
1024² / Euler 8 step / seed 42、CFG=1 / 4 のそれぞれで基準→候補→候補→基準を実行した。
8 走行すべて、同じ CFG の全 denoise step と最終 PNG の SHA が一致した。
しかし全体壁の中央値は CFG=1 が 12.555 → 12.482 秒で中立、CFG=4 が 24.747 → 29.612 秒と悪化した。
単体の合成入力・加重推定を採用根拠にせず、製品の幾何は維持する。
正本は `anima-w4-model.jsonl`、集計は `anima-w4-model-summary.json`。
初回の台本は補助資産 `extras.rope_base` の欠落、次は CFG=1 での negative prompt 指定により拒否された。
これらは実験台本の設定ミスで、ログを保存して修正した後の 8 走行だけを成績に数えた。

### top-p の基数幅の追加試験（不採用）

`radix-width.ts` は既採用の 8 bit × 4 pass に対し、11 bit × 3 pass、16 bit × 2 pass を比較した。
Gemma の実 logits 24 行と 7 サイズ × 4 分布、計 52 入力で候補 id / Float64 確率のビット列が一致した。
温度 0.7 / topP=0.95 / topK 未指定、2 warmup + 7 round、同一入力で順序を反転して測定した。
Gemma 262,144 語の中央値は 11.574 / 10.208 / 9.687 ms だったが、
16 bit は 17 語で 0.00279 → 0.09181 ms と遅く、histogram も 4 KiB → 512 KiB に増える。

さらに 65,536 語以上だけを 16 bit にする `radix-adaptive.ts` を試し、境界の 65,535 / 65,536 / 65,537 語も照合した。
出力は一致したが、共通ループにした実装は小語彙でも約 7〜10% 遅く、大きな Gemma でも
基準 11.520 / 切替 10.426 / 固定 16 bit 9.671 ms と、固定幅の利得を一部失った。
追加のコード複製と小語彙の回帰を引き受けるほどの根拠にはせず、**8 bit × 4 pass を維持する**。
数値の正本は `radix-width.jsonl` / `radix-adaptive.jsonl`。後者の `r11` 欄は切替実装を指す。

### Qwen の GPTQ と短い品質比較

既存の GPTQ 校正・export を Qwen3-0.6B に適用した。16 文 441 token、group=32、
static_groups / act_order 有効、damping=0.01、28 stage / 196 module、約 179 秒で実験資産を書いた。
保存容量は通常の RTN-i4 と同じ **432,956,648 byte**。埋め込み / head は i8、decoder linear は i4 である。
校正文とハッシュ・各層の報告は実験系列の `calibration.json` に保存した。

保存した実バイトを読み戻し、元の周波数を保つ公式 CPU SDPA で 6 文・213 個の次 token 位置を比較した。
正解の文章を毎回入力する teacher-forced 評価であり、生成文の人間評価や一般ベンチマークではない。
NLL は正解の次 token に割り当てた負の対数確率、KL は f32 の出力分布との隔たりで、どちらも小さい方がよい。

| 格納    | 平均 NLL | f32 との平均 KL | f32 と同じ最大確率 token の割合 |
| ------- | -------: | --------------: | ------------------------------: |
| f32     | 3.402840 |               0 |                          1.0000 |
| i8      | 3.418895 |        0.022556 |                          0.9202 |
| RTN-i4  | 3.743736 |        0.434886 |                          0.7136 |
| GPTQ-i4 | 3.671265 |        0.253722 |                          0.7183 |

GPTQ は RTN-i4 に対し KL が約 42% 小さいが、i8 の差には届かず、最大確率 token の一致率の改善も小さい。
6 文からモデル全般の品質を断定しない。本文・token 数・ハッシュは `qwen-quant-quality.json`、
重み復元と既存 golden の atol=1e-3 検査は `qwen-quant-quality-stored.py` に残した。
最初の汎用 safetensors reader は I4 非対応で失敗したため、既存の verify / unpack / dequantize を使って読み直した。

GPTQ の GPU 検証は completion 3 入力 × 8 token に加え、公式 non-thinking chat の 3 問も成功した。
chat の prefill 全 logits の最大絶対差は 0.000245 未満、3 / 10 / 21 token が CPU と厳密一致した。
France は Paris、Japan は「东京」、WebGPU は短い説明だった。通常 i4 の Lyon / 京都から変わるが、小標本である。
`qwen3-06b-gptq-i4-chat-gpu.jsonl` が正本。

同一プロセスの RTN→GPTQ→GPTQ→RTN、各 3 入力 64 token の定常 336 標本ずつでは、
decode 中央値は RTN 21.180919 / GPTQ 22.100336 ms。速度改善ではない。
基準自体も最初の 19.3 ms 前後から最後の 21.5 ms 前後へ変動しているので、差を方式固有の定数としない。
正本は `qwen3-06b-gptq-bench.jsonl`。各入力の初めの 8 token と全 prefill logits は CPU 検証済みで、
残りの token は速度測定用の固定長生成である。

### Chrome と Deno の同一台本での比較

この Linux 環境では Chrome 153 の headless 起動は NVIDIA ANGLE Vulkan を認識しても
WebGPU adapter が null だった。`ForceEnableWebGpuInterop` の追加でも変わらなかったが、
一時的な Xvfb 上の headed 起動で **NVIDIA Ampere / isFallbackAdapter=false** を取得できた。
WebGPU 有効化フラグ付きの検証環境であり、すべてのブラウザ / OS の既定動作を保証する試験ではない。
起動引数と adapter 情報は `browser-diagnose-headed.json` / 各結果 JSON に保存した。

依存バイナリは `/tmp` へ展開した。Xvfb が固定する `/usr/bin/xkbcomp` が存在しなかったため、
一時的な別バイナリの同長文字列だけを `/tmp/xkb` に置き換えた。元バイナリとシステムは変更していない。
両方の SHA と変更内容は `browser-xvfb-local.json` に保存した。外部ソースコードの転記ではない。

同じ `roundTrip()` で 5 warmup + 50 標本を測り、コピーした整数 42 の読み戻しも毎回検証した。

| コピー容量 | Chrome 壁中央値 ms | Deno 2.9.6 壁中央値 ms |
| ---------- | -----------------: | ---------------------: |
| 4 byte     |           0.090000 |              11.344801 |
| 4 KiB      |           0.090000 |              11.359680 |
| 1 MiB      |           2.232500 |              11.620766 |

Deno の小転送にある約 11 ms の床は Chrome にはない。
正本は `browser-headed-roundtrip-qwen3-06b-i8.json` / `deno-roundtrip-matched.json`。

Chrome の Qwen3-0.6B i8 は 3 入力 × 8 token が CPU と一致し、prefill 全 logits の最大絶対差は
0.000144 未満だった（`browser-headed-probe-qwen3-06b-i8.json`）。これは短い数値検証で、正式な速度ベンチではない。

Gemma 4 E2B は同じ HTTP 資産、自由文 prompt、128 token、capacity=8192 で比較した。
各 mode に warmup 1 回 + 測定 2 回、毎回新しい sequence を作る。Deno / Chrome 合計 **18 走行すべての id 列が同一**。
初回 token が届いた後の generation 時間の中央値は次のとおり。

| mode   | Chrome ms |  Deno ms |
| ------ | --------: | -------: |
| plain  |  2747.505 | 3480.547 |
| always |  2795.847 | 3555.984 |
| auto   |  2897.857 | 3694.281 |

この自由文では Chrome の通常生成が約 1.27 倍速いが、投機は通常生成を上回らない。
小さな転送の倍率をモデル全体へ外挿せず、prompt / 受理率が違う課題とも区別する。
正本は `browser-headed-gemma-freeform.json` / `deno-gemma-matched.json`、
集計は `browser-deno-gemma-summary.json`、共通台本は `browser-main.ts`。

### MiniCPM5-2B の GPTQ と短い品質比較

既存の GPTQ 校正を MiniCPM5-2B に適用した。16 文 457 token、group=32、
static_groups / act_order 有効、damping=0.01、42 stage / 294 module を処理した。
校正は 970.405 秒、export と completion 参照まで 1022.019 秒、chat 参照まで 1063.490 秒。
保存容量は通常の RTN-i4 と同じ **1,776,451,608 byte**。実験系列は
`outputs/series/minicpm5-2b-gptq-i4-2026-09-10-probe/`。

保存された量子化重みを復元し、公式 CPU 実装で 6 文・210 個の次 token 位置を評価した。
RoPE の周波数は元のまま保ち、復元後に既存 golden の全 logits も atol=1e-3 / rtol=0 で確認した。
埋め込みと head はそれぞれ 2 つの i8 shard を結合して復元する。

| 格納    | 平均 NLL | f32 との平均 KL | f32 と同じ最大確率 token の割合 |
| ------- | -------: | --------------: | ------------------------------: |
| f32     | 3.069982 |               0 |                          1.0000 |
| i8      | 3.069811 |        0.004278 |                          0.9667 |
| RTN-i4  | 3.292731 |        0.322306 |                          0.7095 |
| GPTQ-i4 | 3.169761 |        0.129570 |                          0.8571 |

この小標本では GPTQ が RTN に対して KL を約 60% 減らし、最大確率 token の一致率も改善した。
i8 と同じ忠実度には達していない。NLL の i8 と f32 の僅差は i8 の一般的優位を意味しない。
正本は `minicpm-quant-quality.json` / `minicpm-quant-quality-summary.json`、
台本は `minicpm-quant-quality-restored.py`。初回の入力パスの組み立てミスは失敗ログに残し、
修正後の出力だけを評価した。校正文と評価文を混ぜず、一般的な言語能力の評価へ外挿しない。

GPU は GPTQ の completion 3 入力 × 8 token、chat 3 問の 2 / 2 / 19 token が CPU と厳密一致した。
prefill 全 logits の最大絶対差は completion 0.000074 未満、chat 0.000212 未満。
RTN-i4 の chat も 2 / 2 / 20 token が CPU と一致し、最大差 0.000328 未満だった。
Chrome でも i8 / GPTQ-i4 の各 3 入力 × 8 token が CPU と一致し、最大差は 0.000100 未満。
正本は `minicpm5-2b-gptq-i4-gpu.jsonl`、`minicpm5-2b-gptq-i4-chat-gpu.jsonl`、
`minicpm5-2b-i4-chat-gpu.jsonl`、`browser-headed-probe-minicpm5-2b-*.json`。

速度は同一プロセスの RTN→GPTQ→GPTQ→RTN、各 3 入力 64 token、
初めの decode 7 回を除いた定常 336 標本ずつで **26.580705 / 26.626631 ms** と同等だった。
GPTQ の利点はここでは容量を変えずに数値の劣化を減らすことにある。
正本は `minicpm5-2b-gptq-bench.jsonl`、集計は同 `-summary.json`。
配布 recipe / 公開 manifest への採用、長文と広い品質評価は別の検収として残る。

### Gemma E4B の全 PLE と既存 pipeline での実行

先の decoder 検証を広げ、42 layer × 256 dim × 262,144 token の PLE（層別の埋め込み）を
すべて用意した。元の checkpoint から 1024 行ずつ読み、既存式の i8 + channel scale で
token 順に 11 shard へ格納した。PLE は **2,862,616,000 byte**、decoder は **3,159,221,160 byte**。
境界 35 点と先の CPU 入力 24 件で、読み戻した PLE が元の量子化計算と厳密一致した。
公式 tokenizer 由来の encode 26 / decode 32 ケースも既存 TS 実装と一致した。

`outputs/series/gemma4-e4b-pipeline-2026-09-11-probe/` に、既存 `karume/4` 形式のローカル manifest を作った。
既存 `Gemma4Pipeline` をそのまま使い、drafter 無し、chunk=16 / maxChunk=64 / capacity=128 とした。
必要な maxBufferSize / maxStorageBufferBindingSize は既存の見積り関数でともに **640 MiB**。
この値は最大の単一バッファに関する要求で、モデル全体の VRAM 容量ではない。

Deno と Chrome の両方で次を確認した。

- completion 3 入力 × 8 token は先の CPU 参照と厳密一致。
- chat は France「Paris」、Japan「東京」、WebGPU の日本語一文が出力され、Deno / Chrome の
  全 id 列と停止理由が一致。本文 token 数は 1 / 1 / 27、EOS は別に数える。
- この走行時点では chat の CPU 参照を未取得だったため、JSON の cpuIdentity は false のまま保存した。
  後段の CPU 照合で確認を追加している。

正本は `e4b-pipeline-deno.json` / `e4b-pipeline-browser.json`、照合は `e4b-pipeline-summary.json`。
資産の内訳は系列内 `pipeline-probe.json`。その finishSeconds / finishPeakRssBytes は manifest を
仕上げた区間だけの値であり、モデル全体の export 時間・最大 RAM ではない。
初回台本は graph shard 単独を verify_model に渡して失敗したため、代表ファイルを渡す形へ修正した。
元の decoder / 公開資産は上書きしていない。

この結果は E4B が既存パイプラインで動く実験資産であり、公開モデル追加の完了ではない。
通常の配布 recipe は E2B と drafter を前提とするため、E4B の正式配布には build の対象分離、
ファミリの source 表、配布 smoke、品質 / 長文の検収が要る。
今回このための公開 API やランタイム変更は行っていない。

### ブラウザでの run 内訳と Fusion の再評価

`browser-phase-profile.ts` は自由文 64 token、各 mode に warmup 1 回 + 測定 1 回で
Session.run / mapAsync / popErrorScope / submit を計測した。Deno と Chrome 合計 12 走行で id 列が一致した。
測定 run の中央値は次のとおり。map は複数出力の待ちが重なるので、最初の発行から最後の完了までを取り、
Promise ごとの待ち時間を足し合わせない。

| 実行環境 / kind               | run ms | map 呼出し前 ms | map 全体 ms | map 同期発行の合計 ms |
| ----------------------------- | -----: | --------------: | ----------: | --------------------: |
| Chrome plain decode（63 本）  | 17.270 |           4.095 |      12.620 |                 0.005 |
| Deno plain decode（63 本）    | 26.252 |           3.003 |      23.064 |                11.982 |
| Chrome always draft（36 本）  |  4.597 |           1.592 |       2.975 |                 0.005 |
| Deno always draft（36 本）    | 14.055 |           1.059 |      12.939 |                 1.858 |
| Chrome always verify（36 本） | 20.322 |           5.227 |      13.195 |                 0.010 |
| Deno always verify（36 本）   | 28.804 |           3.508 |      24.606 |                13.524 |

各列の中央値なので和が run 中央値と一致するとは限らない。
Chrome の popErrorScope 2 本の発行〜最終完了は plain decode 3.685 / draft 1.435 / verify 4.747 ms、
Deno は約 0.005 ms だった。Chrome では GPU コマンドを出した後の検証待ちが見えるが、GPU 計算と重なるため、
この全量を消せる直列費用とみなすのは誤りである。エラースコープは維持し、順序を変える修正は採用しない。
この台本は abort signal 無しで、低遅延 timer の呼び出しは両環境とも 0 件だった。
したがってブラウザの timer clamp に関する仮説は、この測定では検証できていない。
正本は `browser-phase-profile.json` / `deno-phase-profile.json`、集計は `browser-deno-phase-summary.json`。

GPU timestamp を有効にした別走行でも 6 走行の token 列は同一で、全 418 run の timestamp clamp は 0 だった。
plain decode / always draft / always verify の kernel 時間合計の中央値は **19.703 / 3.361 / 21.083 ms**。
plain decode の主な平均内訳は i4 linear 10.503 ms（276 dispatch）、rms_norm 2.160 ms（242）、
strided 1.260 ms（205）、i8 head 1.211 ms（1）。drafter は i8 linear 1.354 ms（68）、
rms_norm 0.445 ms（63）、strided 0.289 ms（48）だった。
測定自体が query と読み戻しを加え、非計測 run より遅くなるので、両者の差を CPU 時間とみなさない。
正本は `browser-gemma-gpu-timed.json`、集計は `browser-gemma-gpu-summary.json`。

Fusion の既存結果 [9/6 の K-15 / K-7 / P-5](2026-09-06-fusion-spikes-k15-k7.md) も再確認した。
今回も linear が大きく、dispatch 本数だけでは融合の利益を判定できない。
新しい候補は小さい RMSNorm / 転置の実形状と消費側をまず絞り、timestamp 無しのモデル全体で判定する。
新 kernel、丸め障壁の削除、検証スコープの省略を、今回の内訳だけで採用しない。

### 次に試作する f16 GEMV の範囲

これは未承認・未実装の提案である。現行の f16 M=1 は共有タイルと barrier を持つ GEMM、
i8 / i4 の小 M は GEMV を使う。Qwen3 の正式比較で f16 の decode が f32 と同程度だったことと、
既存 GEMM のタイル変更が解決にならなかったことから、この実行方式の差を次の候補とする。

試作は `linear-gemv.ts` の f16 格納・M=1 だけに限定する。既存の逐次 K 縮約と最後の bias を保ち、
`unpack2x16float` で読んだ重みを使う。公開 API / manifest / 格納データの変更は含めない。
実装箇所は codegen と linear の族選択、対応するテスト・設計記録である。

1. 実資産の N/K と端チャネル・bias を検収対象として固定する。
2. 実験ディレクトリの切替だけで試作し、同じ key の WGSL の決定性と既存 GEMM との u32 一致を確認する。
3. heater と往復順の単体計測で候補を絞る。
4. Qwen3-0.6B f16 の CPU 参照・生成 id を保って、同一プロセスのモデル A/B を行う。
5. 単体 1.3 倍または対象 decode 全体 10% の改善が再現しなければ採用しない。数値門は緩めない。

大きな dispatch 数を減らせてもコンパイル費が増える可能性があるので、初回と定常を分ける。
RTX での利得を M2 の利得とは扱わない。詳細原稿は `f16-gemv-plan.md` にも保存した。

### QAT mobile の保存重みを部分取得して照合

公式の固定 revision の safetensors からヘッダーだけを範囲取得し、INT2 / INT4 が U8 に詰めて保存され、
INT8 は I8、scale は F32 であることを確認した。ヘッダーは 375,392 byte。
テキスト部分の重み・scale・その他 tensor の payload 合計は **2,118,056,254 byte**。
これは音声・視覚側とヘッダーを除く保存容量で、GPU 常駐容量の実測ではない。
正本は `qat-safetensors-header.json` / `qat-header-summary.json`。

設定の直接比較では、text_config の差は `tie_word_embeddings: true → false` だった。
したがって「同じ構造」は layer / head / 次元などの骨格を指し、重み共有の宣言まで同一ではない。
初期調査の「tied embedding も一致」という記述は訂正する。正本は `qat-config-diff.json`。

ただし、共有が無効なことから重みの値まで異なるとは推論できない。
最初の 64 語を CPU で照合した後、head / embedding の格納整数 **100,663,296 byte ずつ**と
scale **1,048,576 byte ずつ**をストリームで全バイト比較し、双方とも完全一致した。
正本は `qat-tied-bytes-summary.json`。取得した 203,423,744 byte は比較に使い、全量を複製保存していない。
この revision で両者の整数・scale だけを共有するなら、重複する **97 MiB** が削減候補となる。
これは未実装の候補で、別 revision や活性の丸めまで同一として共有することはできない。

さらに最後の MLP down projection の 32 出力行と実 scale を読み、インストール済みの
公式 Transformers `QuantizedLinear` で f32 の合成入力 `[4,12288]` を実行した。
入力 / 出力 scale は 0.0994094536 / 0.1543705314、SRQ を省いた比較では **128 出力すべて**が変わり、
最大絶対差は 0.1191568971 だった。前述の 64 語と合わせた tensor の取得量は 148,112 byte。
これにより、整数を展開するだけで通常 linear へ通す案では公式の演算を再現できないことを具体的に確認した。
台本・取得範囲・SHA・出力 SHA は `qat-small-cpu-probe.py` / `qat-small-cpu-summary.json` に保存した。
公式関数を呼び出しており、外部実装コードは転記していない。
モデル全体の生成・品質・GPU 性能は未検証であり、新しい格納 / SRQ 契約の設計が必要という判断は変わらない。

### 中断境界の timer を CPU 単体で比較

`abort-browser.ts` は既存 `settleAbort` をそのまま呼び、signal 無し / 有り、
連続呼び出し / 別の MessageChannel task の後という 4 条件を各 3 round × 256 標本で診断した。
先に timer で予約した中断を次の境界で拾う確認も、Deno / Chrome 各 25 回成功した。
MessageChannel はこの台本で別 task を挟むためだけに使い、製品の中断待ちを置き換えていない。

Chrome 153 では signal 有りの連続呼び出しが中央値 4.065〜4.090 ms、
別 task の後は 0.005 ms だった。Deno 2.9.6 はどちらも約 2.06〜2.08 ms。
signal 無しは Chrome の時計分解能では中央値 0、Deno は約 0.0003 ms だった。
Chrome は GPU を無効にした headless 起動で、この CPU 診断中には別プロセスの機能検証が進行していた。
したがって厳密な CPU 性能比較ではなく、待ち方の条件差を確認する診断である。

この結果だけからブラウザの生成に毎 token 4 ms を加算してはいけない。
GPU の完了通知など別 task を経由する場合の nesting は、tight loop の連続 timer と異なる。
数値と引数は `browser-abort.json` / `deno-abort-matched.json`、集計は `abort-matched-summary.json`。

### 中断 signal の有無を Gemma の実生成で比較

`browser-abort-model.ts` は同一の自由文 64 token、capacity=8192、新しい sequence ごとの
plain / always を、signal 無し→有り→有り→無しの順で測った。各条件の warmup を先に 1 回ずつ置き、
Chrome / Deno 合計 **24 走行すべての id 列が一致**した。表は warmup を除いた各 2 走行の中央値。

| 実行環境 / mode | signal 無しの生成 ms | signal 有り ms | 有り時の timer 中央値 ms |
| --------------- | -------------------: | -------------: | -----------------------: |
| Chrome plain    |             1345.695 |       1349.945 |                    0.010 |
| Chrome always   |             1286.595 |       1286.737 |                    0.015 |
| Deno plain      |             1728.041 |       1865.491 |                    2.084 |
| Deno always     |             1619.639 |       1701.253 |                    2.090 |

実生成では Chrome の連続 timer の 4 ms は再現せず、signal の有無は中立だった。
Deno は plain 約 8% / always 約 5% の追加時間があり、単体で観測した約 2 ms の timer 待ちと整合する。
1 走行の timer 本数は plain 64 / always 37。投機は複数 token をまとめて確定するため境界の数も少ない。
中断を受け付けるために必要な境界を間引く修正は入れない。

これは 64 token の出力区間であり、前段の 128 token 比較と受理分布が違う。
この短い区間では always が plain より速いが、その比率を自由文全体や長い継続へ外挿しない。
正本は `browser-abort-model.json` / `deno-abort-model.json`、集計は `abort-model-summary.json`。

最初の Chrome 起動は Vulkan の feature 指定が漏れて adapter が null だった。
成功済みの `--enable-features=Vulkan` を含む引数へ揃えると完走した。失敗ログも保持し、成績から除外した。
使用した Puppeteer は enable-features を内部で統合するときに入力 args 配列から削除するため、
起動後の args を保存した JSON だけではこの flag を再現できない。実際の起動コードを
`browser-drivers/` へ保存した。今後は渡す前の配列または実プロセスの spawnargs を記録する。

### E4B 会話の CPU 照合を追加

`e4b-chat-cpu.py` は同じ公式 checkpoint から量子化済み `ProductChunkWrapper` の CPU 参照を再構築した。
KV cache を使わず、各 token の全 prefix を f32 で計算する。
先に保存済み completion 3 ケースの prefill を再計算し、logits / hidden とも最大絶対差 0 を確認した。

その参照で France / Japan / WebGPU の chat を生成すると、**EOS 込み 2 / 2 / 28 token** が
Deno / Chrome の両方と厳密一致した。WebGPU の最小 top-1 / top-2 margin は 0.0993042 だった。
CPU の実行時間は 116.482 秒、プロセスの最大 RSS は 27,012,087,808 byte。
この RSS は CPU 参照の構築・計算であり、ブラウザでの推論に必要な容量ではない。

正本は `e4b-chat-cpu/summary.json` と各問の prefill tensor、実行間の照合は
`e4b-pipeline-cpu-summary.json`。元の GPU 実行 JSON にある cpuIdentity=false は、その時点の記録として残した。
E4B の短い会話についても CPU / Deno / Chrome の一致を確認できたが、広い品質評価と長文は引き続き未検証である。

## 追加 LLM のローカル CLI（2026-09-11）

利用者の依頼により、[MiniCPM5-2B](../../examples/minicpm5/README.md) と
[Qwen3-0.6B](../../examples/qwen3/README.md) を試せる CLI を追加した。
取得元を省略すると既存のローカル系列を優先し、現在の手元では GPTQ i4 を選ぶ。
Anima / Irodori の既定もローカル配布形を優先する（`acee48e`）。Gemma 4 は元からローカルが既定。

新 CLI は単発・greedy・非 thinking。入力文と system 指示、文章継続、標準入力、逐次復号、
EOS 停止、JSON 出力を持つ。KV 容量 128 / prefill 64 行の既存実験グラフを使い、
`入力 token 数 + max-new-tokens − 1 ≤ 128` を GPU 重み転送前に検査する。
公開 pipeline / 配布 recipe / source 表の追加や、長文対応はこの変更に含めない。

実行時の追加依存は無い。既存の BPE と Qwen の文字走査を使い、MiniCPM の数字 3 文字分割・
BOS と、各モデルのチャット形式を examples 内で結線した。Unicode 表は既存の生成器から作成し、
元の tokenizer と NFC（Unicode 正規化）の 12,232,704 文脈 + 乱択 2,000 件を照合した。
両モデルで入力・復号 73 ケース、チャット 15 ケースずつが公式 tokenizer と厳密一致する。
表と参照列は次の手順で別ディレクトリへ再生成できる（実行時に Python は不要）:

```sh
tools/.venv/bin/python examples/shared/emit-llm-fixtures.py --out /tmp/karume-llm-fixtures
```

保存結果の所在は `outputs/bench/karume/2026-09-11_cli-local-models/`:

| 検証                                           | 結果                                                               | 生データ                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| GPTQ i4 の CLI: 2 モデル × チャット 3 + 継続 1 | 全 8 件で CPU 参照の token 列・本文と一致。チャットは EOS まで一致 | `cli-smoke-summary.json` / `check-cli.py` / 各 CLI の JSON・stderr |
| 標準入力からの MiniCPM5 日本語生成             | 逐次 stdout が `東京` + 改行                                       | `stdin-minicpm5.stdout.txt` / `cli-options.log`                    |
| Qwen3 の `--quant i8`                          | 選択先と継続 8 token が CPU 参照と一致                             | `quant-i8-qwen3.json`                                              |
| 容量超過                                       | 重み転送前に終了コード 1 で拒否                                    | `capacity-rejection.stderr.log`                                    |
| Unicode 表・参照列の再生成                     | 同じデータを再現                                                   | `fixture-regeneration.log` / `regenerated/`                        |
| ローカル優先の全体検証                         | 2,822 tests / 743 steps、失敗 0、ignore 5                          | `verify-local-first.log`                                           |
| CLI 追加の全体検証                             | fmt / lint / check / test のログ                                   | `verify-llm-cli.log`                                               |

RTX 3080 Ti で実施。モデルの内容の正確さを認定する検証ではない。
例えば Qwen3 の日本語回答は CPU 参照も `日本の首都は、**东京**です。` と出す。
長文・多ターン・モデル全般の品質評価・M2 での実行は引き続き未検証。

## 追加 LLM CLI の多ターン対応（2026-09-11）

利用者の依頼により、f16 GEMV / QAT mobile INT2 に先立って MiniCPM5 / Qwen3 の対話を実装した。
既存グラフと実行 API で成立するため、runtime カーネルや公開 pipeline の追加は要らない。
`demo:gemma4` と同様に行単位の会話、`/reset`、`/exit`・`/quit`・EOF、生成中の SIGINT を扱う。
`--prompt` は単発実行、`--completion` の標準入力は EOF まで読む従来の文章継続を維持する。
対話の `--json` は回答ごとに JSON 1 行を出し、操作案内は stderr へ送る。

重みの Session は会話中に保持する。KV キャッシュを継ぐのは EOS で閉じたターンだけで、
次の公式テンプレートの token 列と commit 済みの列が完全一致することを条件とした。
最後に配送した token は未 commit なので、次の prompt の差分から入力する。
Qwen は履歴の assistant から空 thinking block を外すため、通常の次ターンで prefix が変わる。
MiniCPM は過去の空 block も維持する。一律の差分連結は Qwen の公式入力と一致しないため採らない。
BPE 再符号化で prefix が変わる場合も同じ検査で再構成する。rewind は使わない。

容量は既存どおり 128 token、prefill は 64 行。生成予算を含めて収まらなければ古い質問と回答の
対を削り、件数を通知する。system と今回の質問は必ず残す。今回の質問だけでも大きすぎる場合は
元の履歴を変更せず拒否し、次の入力を受ける。中断・生成上限で切った回答は表示済みの本文を
履歴に残し、未閉鎖の KV は返却する。未出力で中断した質問は履歴に残さない。

生データ・再現スクリプトの所在は `outputs/bench/karume/2026-09-11_llm-multiturn/`:

| 検証                      | 観測結果                                                                          | 記録                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 公式 tokenizer の多ターン | 両モデル各 24 件で厳密一致。既存 73 入力・復号 + 15 単発 chat と Unicode 表は不変 | `fixture-generation.log` / `fixtures/` / `unit.log`                                              |
| 実 GPU と公式 CPU の会話  | 両モデル各 7 回、計 14 回で入力・生成 ID・本文・停止・履歴削除件数が一致          | `reference.py` / `reference.log` / `gpu-parity-summary.json` / 各 `*-cli.jsonl`                  |
| SIGINT 後の復帰           | 両 CLI に実際に SIGINT を送り、次の質問・reset・終了まで成功                      | `check-interrupt.py` / `interrupt-summary.json` / 各 `*-interrupt.stdout.txt`                    |
| 大きすぎる質問の拒否後    | 両 CLI で直前の履歴を保持し、次の回答も CPU と一致                                | `check-overflow.py` / `overflow-summary.json`                                                    |
| 単発の回帰検証            | 従来のチャット 3 + 継続 1 × 2 モデル、全 8 ケースで CPU と一致                    | `check-single.py` / `cli-smoke-summary.json`                                                     |
| キャッシュ比較            | 毎回 token 列を CPU 参照と照合。条件・数値は次表                                  | `cache-bench-inclusive.ts` / `cache-bench-inclusive.json` / `cache-bench-inclusive-summary.json` |
| 全体検証                  | fmt / lint / check / test の実行ログ。完了結果は同ディレクトリの最終状態記録      | `verify.log` / `FINAL-STATUS.json`                                                               |

CPU 参照は保存済み GPTQ i4 パラメータを公式モデルへ復元し、全パラメータを読んだことと
従来の全 logits golden（atol=1e-3 / rtol=0）への一致を先に検査した。
RoPE の inv_freq は公式 f32 のまま保持し、各ターンを公式 tokenizer / SDPA で独立再計算した。

RTX 3080 Ti、各方式 1 会話 warmup 後に cache / replay の順を交互に 4 回ずつ計測した中央値。
前ターンのキャッシュ返却を両方式の計測に含め、重みの読み込みは含めない。
単位は ms、入力処理は最終 prefill 完了まで、全体は EOS を含む全 token の配送と終了まで。

| モデル・ターン | 毎回再計算: 入力 / 全体 | 条件付き再利用: 入力 / 全体 | 再利用 token |
| -------------- | ----------------------- | --------------------------- | ------------ |
| MiniCPM5・2    | 103.67 / 133.35         | 84.19 / 111.29              | 28           |
| MiniCPM5・3    | 200.52 / 232.39         | 83.92 / 111.33              | 57           |
| Qwen3・2       | 101.91 / 201.50         | 104.22 / 192.75             | 0            |
| Qwen3・3       | 187.06 / 212.36         | 186.74 / 210.83             | 0            |

MiniCPM の 3 ターン目は prefill が 2 chunk から 1 chunk へ減り、入力処理が約 58% 短縮した。
Qwen は両方式とも再計算であり、高速化を主張しない。最初の `cache-bench.*` は replay 側だけ
reset を計測外にしていたため比較値に採用せず、計測境界を修正した `*-inclusive.*` を正本とした。

これは短い既定 greedy 会話の一致と操作の検収であり、モデル全般の回答品質を保証しない。
例えば Qwen は国名を求めた 5 回目にも `Tokyo.` と答える（CPU も同じ）。長文・公開配布・
M2 / ブラウザでの今回の多ターン操作、f16 GEMV / INT2 の試作は未完のまま別項目として残す。

## f16 格納 M=1 の GEMV（2026-09-11）

利用者の優先指示に従い、Sol 3 名の序盤調査後は主担当だけで試作・検証した。
コンテキスト容量の拡張は後回し。実測は RTX 3080 Ti / Deno 2.9.6。
保存先は `outputs/bench/karume/2026-09-11_optimization-next/`。
この節は f16 の採否を扱い、INT2 / SRQ と Anima の追加候補は別の調査単位とする。

### 適用範囲と採用判断

既存 f16 は M=1 でも共有タイルと barrier を持つ GEMM を使っていた。
既存 GEMV 族へ f16 の読み出しを追加し、1 スレッドが 1 列を K 昇順に計算する。
`unpack2x16float`、`acc = acc + x * w`、最後の bias 加算を維持する。
門は M=1・f32 計算・K>0・K%8=0・N%4=0。公開 API / IR / 資産は無変更。
判断は [ADR 0082 追記 6](../decisions/0082-linear-gemv-decode.md#追記-62026-09-11-f16-格納の-m1-を-gemv-族へ広げるk-25)。

実 Qwen の f16 linear は 197 本、形状 `(N,K,本数)` は
`(2048,1024,28)` / `(1024,1024,56)` / `(1024,2048,28)` /
`(3072,1024,56)` / `(1024,3072,28)` / `(151936,1024,1)`。
全て適格。Gemma の既存 i4 / i8 資産と MiniCPM5 の現行 f32 / 整数量子化資産はこの変更の対象ではない。

### 単体と実モデルの結果

`f16-micro-focused.jsonl` は正の K を持つ 14 形 × 6 腕で全出力 u32 一致。
実形状 9 形は heater を挟み各 5 回の最小 GPU 時間を採り、5 形は端の一致だけを検査した。
順序は GEMM → c32u4 → c32u16 → c32u16 → c32u4 → GEMM。
Qwen の層数で重み付けし、各腕の最小値を往復 2 腕で平均した合計は
GEMM **25.650 ms**、c32u4 **5.948 ms（4.31 倍）**、c32u16 **4.802 ms**。
単体は合成重みを使用し、実生成の壁時計とは区別する。

`f16-model-wall.jsonl` は timestamp と logits hash 計算を使わない別走行。
同一プロセスの同じ往復順で、3 入力をそれぞれ 64 token 生成した。
decode 63 回の先頭 7 回を除いた 56 回の中央値を求め、往復 2 腕の中央値を掲載する。

| 入力       | GEMM ms/token | c32u4 ms/token | c32u16 ms/token | 採用 c32u4 の倍率 |
| ---------- | ------------: | -------------: | --------------: | ----------------: |
| capital-en |        42.992 |         22.581 |          23.620 |              1.90 |
| capital-ja |        42.974 |         21.932 |          21.531 |              1.96 |
| webgpu     |        42.649 |         21.750 |          21.230 |              1.96 |

c32u16 は入力により勝敗が変わり、head 単体も c32u4 の 0.962 ms に対して 1.246 ms。
安定した上積みの根拠がないため既存の c32u4 を採用する。
当初の打ち切り線「単体 1.3 倍または decode 全体 10%」を両方満たす。

初期化の観測は単体 JSON の `compileMs` にあるが、この値は Session 作成から初回 run 完了までであり、
シェーダのコンパイル時間だけではない。最初の形の初回は GEMM 71.4 ms / c32u4 113.4 ms
（`f16-micro.jsonl`）。全体の prefill は M=16 で今回の変更対象外。
初回準備と定常 decode を混ぜて倍率を出さない。

### 数値と境界の検証

- `f16-model.jsonl`: 同じ 6 腕 × 3 入力で CPU 参照 logits（既存 atol=1e-3 / rtol=0）と
  既存 greedy 列を維持。各 decode の全 logits を SHA-256 で比較し、最初の GEMM 腕に対する
  **945 回の比較が全て一致**。出力 token だけの比較ではない。
- 採用する製品 WGSL と、モデルで測った c32u4 試作 WGSL はバイト一致。
- `f16-codegen.log`: 82 tests 成功。既存 snapshot を変更せず f16 snapshot を追加。
- `f16-gpu-test.log`: 追加 3 tests 成功。8 形の通常 GEMM との u32 一致と CPU 参照、
  M / K / N / 格納を 1 条件ずつ外した 4 形、f16 計算の指定を検証。
- `f16-mutation.log`: f16 の対の積和順だけを反転した試作で u32 アサーションが失敗。
  検査を通すための期待値・許容差の変更はしていない。

初期掃引は最後に加えた K=0 の **GEMM 比較側**で `Binding size 4 … less than minimum 16` により停止した。
これは既存の limitations・recipe-builder のコメント・gpu_i8a8_test が記録する非対応域で、新規回帰ではない。
最初に想定外と報告したが、照合後に訂正した。`K0-STOP.md` と元ログを保存し、K=0 自体の修正は混ぜない。
新しい f16 の門は K>0 に限定し、既存テストは維持した。

### Chrome の追試と全体検証

Chrome 153.0.8010.36（Vulkan / NVIDIA Ampere、fallback adapter=false）でも同じ台本を実行した。
`browser-f16-parity.json` は GEMM → c32u4 → c32u4 → GEMM、3 入力 × 64 token。
CPU 参照と既存 greedy 列を維持し、最初の GEMM 腕に対する **567 回の全 logits SHA 比較が一致**。
`browser-f16-wall.json` は timestamp / hash 計算を外した別走行で、代表値の取り方は Deno と同じ。

| 入力       | GEMM ms/token | c32u4 ms/token | 倍率 |
| ---------- | ------------: | -------------: | ---: |
| capital-en |        20.430 |         11.899 | 1.72 |
| capital-ja |        20.501 |         11.751 | 1.74 |
| webgpu     |        20.854 |         11.856 | 1.76 |

集計は `browser-f16-summary.json`。wall の console に残る固定 `bitEqual` 表示は根拠にせず、
JSON の `checkBits=false` と、別の parity 走行を区別する。
Deno と Chrome の絶対時間の差を、そのまま TypeScript の費用とは帰属しない。
Apple M2 は未検証。RTX での成績を M2 へ外挿しない。

`deno task verify` は **2,843 passed / 743 steps / 0 failed / 5 ignored、24m38s**。
`f16-verify.log` と `f16-final-status.json` に保存する。Chrome 追試後の文書更新は fmt と差分を再確認した。

## f32 格納 M=1 の GEMV（2026-09-11）

f16 のコミット `f8bbaf1` 後、主担当が独立試作して採否を検証した。
保存先は前節と同じ `outputs/bench/karume/2026-09-11_optimization-next/`。
RTX 3080 Ti / Deno 2.9.6。候補の打ち切り線は単体 1.3 倍または decode 全体 10%。
公開 API / IR は追加せず、重みを `vec4<f32>` で読む既存族の変種を追加する。
M=1・f32 計算・K>0・K%4=0・N%4=0 に限り、K 昇順の積和と最後の bias を維持する。

### 単体と Deno の実モデル

`f32-micro-focused.jsonl` は GEMM → GEMV → GEMV → GEMM の順で 16 形、64 件が u32 一致。
実形状 11 形は heater を挟んで 5 回の最小 GPU 時間を採り、端の 5 形は一致だけを検査した。
Qwen は前節と同じ 197 linear。MiniCPM5 は 295 linear で、IR の現物から確認した `(N,K,本数)` は
`(2048,2048,84)` / `(256,2048,84)` / `(6144,2048,84)` / `(2048,6144,42)` /
`(130560,2048,1)`。どちらも全 linear が適格。

往復 2 腕の最小時間を平均して層数で加重した合計は、Qwen **27.700 → 11.552 ms**、
MiniCPM5 **74.416 → 34.229 ms**。単体は合成重みであり、モデル全体の時間とは区別する。
中間層の改善に比べ、head の単体改善は Qwen 1.05 倍、MiniCPM5 1.18 倍と小さい。
集計は `f32-micro-summary.json`。`compileMs` は Session 作成から初回 run までの観測であり、
シェーダのコンパイル単独の値ではない。

実モデルは既存 f32 資産で同じ往復順、3 入力を各 64 token 生成した。
`f32-{family}-parity.jsonl` では既存 CPU 参照の全 prefill logits（atol=1e-3 / rtol=0）と
既存 greedy 列を維持。各 decode の全 logits SHA を最初の GEMM 腕と比較し、
**各モデル 567 回、計 1,134 回が一致**した。これは短い固定入力での数値検収であり、
モデル全般の品質・長文の検証ではない。

`f32-{family}-wall.jsonl` は timestamp / hash 計算を使わない別走行。
decode 63 回の最初の 7 回を除き、残り 56 回の中央値を採り、往復 2 腕の中央値を掲載する。
集計コードは `summarize-f32.py`、結果は `f32-model-summary.json`。

| モデル・入力         | GEMM ms/token | GEMV ms/token | 倍率 |
| -------------------- | ------------: | ------------: | ---: |
| Qwen・capital-en     |        42.658 |        27.322 | 1.56 |
| Qwen・capital-ja     |        42.607 |        27.376 | 1.56 |
| Qwen・webgpu         |        42.647 |        27.506 | 1.55 |
| MiniCPM5・capital-en |        88.798 |        53.405 | 1.66 |
| MiniCPM5・capital-ja |        88.876 |        53.188 | 1.67 |
| MiniCPM5・webgpu     |        89.282 |        53.094 | 1.68 |

### 適用と検証

- 採用は c32u4。製品コードと実測した試作の WGSL は f32 / f16 / i4 / i8 の全変種で一致する。
- `f32-targeted-test.log`: codegen と f16 / f32 の GPU 境界テスト、計 89 tests 成功。
  f32 の 8 形は通常 GEMM の先頭行との u32 一致と CPU 参照を検査し、M / K / N の各門を外した
  3 形と f16 計算の指定も検査する。既存 snapshot の変更はなく、f32 を追加した。
- `f32-mutation.py` / `f32-mutation.log`: f32 の隣接成分 2 個の積和順だけを入れ替え、
  u32 比較で拒否することを確認した。許容差や期待値は緩めていない。
- Apple M2 は未検証。Deno と Chrome の差を TypeScript の費用だけには帰属しない。

### Chrome の追試

Chrome 153.0.8010.36、Vulkan / NVIDIA Ampere、fallback adapter=false。
両モデルで同じ往復順と入力を使い、parity と wall を別走行にした。
各モデル 567 回、合計 1,134 回の全 logits SHA 比較が一致し、既存 CPU 参照と greedy 列も維持した。
壁時計の集計方法は Deno と同じ。

| モデル・入力         | GEMM ms/token | GEMV ms/token | 倍率 |
| -------------------- | ------------: | ------------: | ---: |
| Qwen・capital-en     |        22.551 |        17.081 | 1.32 |
| Qwen・capital-ja     |        22.476 |        17.091 | 1.32 |
| Qwen・webgpu         |        22.450 |        17.044 | 1.32 |
| MiniCPM5・capital-en |        53.961 |        42.435 | 1.27 |
| MiniCPM5・capital-ja |        54.071 |        42.647 | 1.27 |
| MiniCPM5・webgpu     |        54.042 |        42.582 | 1.27 |

`browser-f32-{family}-{parity,wall}.json` に保存し、集計は `browser-f32-summary.json`。
GPU カーネルの利得と、環境により違う待ち時間を混ぜて外挿しない。
全体検証のログは `f32-verify.log`、完了状態は `f32-final-status.json` に記録する。

`deno task verify` は **2,847 passed / 743 steps / 0 failed / 5 ignored、24m42s**。
検証開始時に記録したコード SHA は、完了後も全て不変だった。

## QAT mobile の INT2 と固定丸め（2026-09-11）

f32 のコミット `eaccc9b` 後に主担当が行った独立実験。保存先は
`outputs/bench/karume/2026-09-11_optimization-next/`。RTX 3080 Ti / Deno 2.9.6。
この節は単体の実測と製品統合前の提案であり、公開 IR / runtime / exporter の対応を意味しない。

### 公式形式と既存形式の違い

対象は [Google の QAT mobile モデル](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers)。
通常 E2B と骨格は同じでも、固定量子化された重みと途中の丸め処理が異なるため、別の量子化 profile として扱う。
取得 revision、Range、SHA は `qat-full-projections/manifest.json` と旧調査の
`outputs/bench/karume/2026-09-10_model-optimization/qat-safetensors-header.json` に保存した。
外部実装の複製はせず、公式 Transformers の CPU 関数を参照計算として呼び出した。

`qat-module-census.json` は公式 config、meta device 上の全 module、保存済み tensor header を照合した結果。
text linear は INT2 **61 本**、INT4 **145 本**、INT8 **70 本**。
token embedding は INT2、PLE（層別の埋め込み）は INT4。U8 という物理 dtype だけでは bit 数を決められない。
INT2 は下位 2 bit から 4 値/byte、格納値 0..3 を整数 -2..1 として復元し、linear は出力行ごとの scale を掛ける。
SRQ は固定 scale による丸めで、scale=0 は恒等、それ以外は
`clamp(round_even(f32(x / scale)), -128, 127) * scale`。既存の動的 absmax の w4a8 とは異なる。

INT4 の固定重みは再量子化せず、既存の group 形式へ scale を反復できる。
`qat-i4-normalize/summary.json` では up の実 8 行と PLE の実 2 行を既存 pack/unpack で往復し、
公式の f32 復元値と全 u32 一致。q=-8 も含む。全 text の INT4 で scale 反復の追加量は
**2,420,736 bytes**。製品化では既存 helper の記述と実装の符号域を明示的に整える必要がある。
ただし `packages/models/src/gemma/ple.ts` の sidecar reader は I8 専用で、
通常 embedding が I4 を読めることは PLE の対応を意味しない。専用の宣言・行 byte 範囲・全量/区間読取の変更が必要。

### SRQ の境界差を特定した

CPU の丸めを単純に WGSL へ移すと、down の境界入力 49,152 個中 **5,693 個**で SRQ が異なった
（`qat-division.json`）。CPU の商 -123.5 に対して GPU が -123.49999237060547 となる例があり、
偶数丸めを選ぶだけでは解消しない。
[WGSL の精度仕様](https://www.w3.org/TR/2026/CRD-WGSL-20260831/#floating-point-accuracy)は、
分母の絶対値が 2^-126 以上 2^126 以下の f32 除算に 2.5 ULP の誤差を許す。

独立試作では scale ごとに 128 個の f32 閾値（512 bytes）を CPU で生成し、
GPU の除算は整数区間の候補にだけ使い、丸めの最終判定は入力と閾値の比較で行った。
閾値は「f32 除算の丸め後に偶数丸め」という二段階の境界から導出する。
詳細は `QAT-SRQ-PROOF.md`、実装は `qat-thresholds-v2.py` / `qat-gpu-v2.ts`。
実 input/output scale と 0.125、1 の 4 scale について、閾値の両隣・正負・±0・乱数を含む
**803,080 入力が CPU / GPU とも公式参照に u32 一致**した。
これは試した scale と RTX の検証であり、全 scale・非有限値・subnormal・別 GPU の保証ではない。
最初の閾値生成試作の失敗は NumPy scalar 比較の暗黙変換が原因で、v2 で修正し元ログも残した。

追加の `qat-srq-scale-sweep-{cpu,gpu}.json` は seed=60911、1,031 scale（0 の恒等を含む）、
**857,792 入力で CPU / GPU とも全 u32 一致**。非ゼロ scale は約 2.39e-10〜4.27e9 で、
非ゼロ scale ごとに全 128 閾値と両隣・正負・±0・乱数を検査した。0 は恒等変換を検査した。
元の 4 scale 試験とは入力の重複がある。
これは有限値の追加検証で、NaN / Inf / subnormal と M2 は未検証のまま。

### 実形状の packed INT2 と既存カーネルの比較

最後の MLP の down / up と語彙 head の実重みを取得し、同じ整数と scale を
f32 / I4 / I8 に損失なく展開して比較した。別モデルの重みを比べた数値ではない。
`qat-existing-storage.py` は可逆変換の照合と SHA を保存する。
INT2 試作は 32 スレッド、K 昇順の積和、unroll 1/2/4。N/K は既存版と同じ uniform に渡す。
初版の INT2 だけ N/K を定数化した比較は公平性が不足したため、後続 v2 と最終版で再測した。

`qat-packed-current-baselines.json` の 105 記録は、数値確認 63 件と時間計測 42 件。
3 形 × 3 入力で、f32 GEMM を基準とした他 6 方式の **54 比較が全出力 u32 一致**した。
時間は入出力 SRQ を含む 3 pass、20 反復、heater 後の 5 samples の最小値を採り、
7 方式を正順・逆順で測った 2 値の平均。集計は `summarize-qat-final.py` / `qat-packed-summary-final.json`。

| 形状 (N,K)         | f32 GEMV ms | I4 GEMV ms | I8 GEMV ms | INT2 u4 ms | 対 I4 倍率 |
| ------------------ | ----------: | ---------: | ---------: | ---------: | ---------: |
| down (1536,12288)  |       0.515 |      0.181 |      0.264 |      0.140 |       1.29 |
| up (12288,1536)    |       0.219 |      0.056 |      0.072 |      0.047 |       1.18 |
| head (262144,1536) |       4.379 |      0.711 |      1.029 |      0.308 |       2.31 |

INT2 u1/u2/u4 は down 0.154/0.136/0.140 ms、up 0.048/0.048/0.047 ms、
head 0.322/0.316/0.308 ms。u4 は比較用の代表で、全モデルで採用する設定は未決定。
小さな MLP 重みが GPU cache に収まる条件だけで判断しないため、head の全行も測った。
head の重み＋scale は INT2 **101,711,872 bytes（97 MiB）**、I4 **204,472,320 bytes**、
I8 **403,701,760 bytes**、f32 **1,610,612,736 bytes（1.5 GiB）**。
モデル全体の VRAM ピークや生成速度の実測ではなく、その倍率へ外挿しない。
`compileMs` は createComputePipelineAsync の時間で、後続形は同一コードの cache が温まっている。

### CPU との差と検収の限界

全行の比較では CPU と GPU の線形縮約順に由来する差が残った。
`qat-full-attribution.json` では GPU の丸め前の値を公式 CPU SRQ へ渡すと、全 9 入力で GPU 最終値と u32 一致。
残差を SRQ の実装誤りと混同しない。

- down の sin と up の境界入力は各 1 出力で ±0 のみが異なる。
- up の wide は 1/12,288 出力で隣の量子化区間へ移る。
  index 11381 の丸め前は CPU 4.922986507 / GPU 4.922981262、
  最終値は CPU 4.942914963 / GPU 4.903052330。小さな縮約差が固定丸めで拡大する。
- head は scale=0 で通常の f32 縮約差が残り、sin/wide の最大絶対差は 4.53e-6 / 1.45e-4。
  head の near-boundary ケースは scale=0 から作ったゼロ入力であり、境界の根拠には数えない。

期待値・許容差は変更していない。モデル全体の logits、生成列、品質の CPU 一致は未検証。

### 製品化の段階案（判断待ち）

既存形式の改善だけを進める案は公開契約を増やさないが、QAT の容量利得を使えない。
単体では既存 I4 を上回る利得を確認できたため、**INT2 と固定 SRQ を明示した段階的対応を推薦**する。
ただし公開形式・新カーネル・PLE reader にまたがる変更なので、次の範囲を判断対象とする。

1. IR の i2 と固定 SRQ の契約を定義。packing、scale=0、丸め、有限値の範囲、旧 reader との版互換を明示する。
   INT4 は固定値を維持する group 正規化を使い、再量子化で黙って値を変えない。
2. loader / CPU 参照 / GPU に実装し、単行 GEMV に加え prefill の GEMM と embedding を検証する。
   packing と SRQ と縮約差を個別に検査し、既存の許容差を緩めて通さない。
3. Gemma QAT 専用 recipe と PLE の INT4 宣言・読取を対応する。既存 I8 reader は維持する。
   PLE を I8 に広げる案は重みが倍になるため、容量利得と引き換えであることを明示する。
   exporter / recipe の両 pytest と deno verify を実行する。
4. 通常生成から CPU、Deno、Chrome の複数入力で検収し、RAM / VRAM ピークと中断・解放を確認する。
   M2 は別実機で検証する。SRQ の単体一致だけで全モデルの品質を判断しない。
5. head と token embedding の共有を検討する。固定 revision の整数・scale 全 byte の一致は
   旧 `outputs/bench/karume/2026-09-10_model-optimization/qat-tied-bytes-summary.json` と今回の取得 SHA で照合済みだが、97 MiB の節約候補を全体計測で検収する。
   PLE は別物。MTP の現行 drafter が借用する I8 重みの変更は、通常生成の検収後に別途設計する。

この調査単位では製品コードを変更せず、K-27 を「単体試作済み・統合案の判断待ち」として残す。

この調査コミット前の `deno task verify` は **2,847 passed / 743 steps / 0 failed / 5 ignored、24m44s**。
ログは `qat-verify.log`。追加 SRQ の GPU 照合は全体検証の終了後に直列実行した。

## Anima の RMS 正規化と並べ替え融合（2026-09-11）

QAT の記録を `cbd0b8c` にコミットした後、主担当だけで調査した。
保存先は `outputs/bench/karume/2026-09-11_optimization-next/`。RTX 3080 Ti / Deno 2.9.6。
RMS 正規化は、各行の二乗平均を使って値の大きさを揃える処理。

### 単体で効果を分離した

既存 census の Anima / transformer / 1024px には、permute → rms_norm が 112 対あり、
全て permute の出力の消費先は 1 個だった（`anima-permute-rms-census.json`）。
84 対は `[1,4096,16,128] → [1,16,4096,128]`、28 対は T=512 の同じ並べ替え。
独立試作では、融合によって入力の基点だけを変える案と、RMS のスレッド数だけを減らす案を分けた。

`anima-permute-rms.json` は実 2 形と端 2 形、28 記録。
各形の最初の通常版に対する **24 比較が全出力 u32 一致**した。
実 2 形は 7 heated samples の最小 GPU 時間を取り、正順・逆順の 2 値を平均した。
入力・weight は合成値。端は `[2,13,3,17]` と `[1,1,2,257]` で、後者は 128 版の対象外。
大きい形では 65,536 行を 65,535 workgroups で処理し、行の再訪も検査した。

| T    | 通常 256 ms | 通常 128 ms | 融合 256 ms | 融合 128 ms |
| ---- | ----------: | ----------: | ----------: | ----------: |
| 4096 |       0.509 |       0.352 |       0.484 |       0.239 |
| 512  |       0.078 |       0.044 |       0.068 |       0.037 |

通常版は permute と RMS の両 dispatch を含む。同じ pass の timestamp で計測する。
往復間の揺れもあるため、元の samples を `anima-permute-rms.json` に残した。
本数で重み付けした合計は通常 256 **44.892 ms**、通常 128 **30.812 ms**、融合 128 **21.105 ms**。
単体の加重合計であり、モデル全体の実測ではない。集計は `anima-permute-rms-summary.json`。

### 128 スレッド版を画像生成全体で確認した

通常の並べ替えを残したまま、幅 128 以下の RMS だけを 128 スレッドへ変更した。
上半分のゼロ加算を 1 段省き、残りの加算順序を維持する。
対象は `anima-v1.0`、`f16+dit8-a8-attn8-s16`、1024×1024、8 steps、seed 42、Euler。
入力は `1girl, solo, upper body`、CFG=4 では negative に `low quality`。
CFG は条件の影響を強める係数で、1 と 4 の両方を検査した。

`anima-rms-model-parity.jsonl` は各 CFG で通常 → 128 → 128 → 通常の順。
各 8 ステップの潜在値 SHA と最終 PNG SHA は、最初の通常版に対して全て一致した
（潜在値 **48 比較**、PNG **6 比較**）。この走行の時間は hash 回収と初回準備を含むため性能値には使わない。

`anima-rms-model-wall.jsonl` は別走行。各 CFG の両方式を先に 1 生成ずつ暖機し、
ステップごとの hash を外して同じ往復順で測った。PNG の一致は計測窓の後で確認する。
表は往復 2 走行の平均。集計は `anima-rms-model-summary.json`。

| CFG | 通常 全体 ms | 128 全体 ms | 全体短縮率 | 通常 transformer ms | 128 transformer ms |
| --- | -----------: | ----------: | ---------: | ------------------: | -----------------: |
| 1   |   12,521.928 |  12,320.197 |      1.61% |           9,445.883 |          9,229.019 |
| 4   |   20,880.161 |  20,656.344 |      1.07% |          17,652.366 |         17,421.454 |

大幅な改善ではないが、単体 20% と全体 1% の採用線を満たすため既存 RMS 族の 128 版を採用する。
判断は [ADR 0017 追記 1](../decisions/0017-rms-norm-conv2d-clamp-min.md#追記-12026-09-11-幅-128-以下の-rms-正規化は空の上半分を省くk-28)。
公開 API / IR / 資産に変更はない。Anima の画像品質改善を主張するものでもない。

並べ替えまで融合する追加効果は、通常 128 との差が単体加重で約 9.707 ms / forward。
この値を今回の 8/16 forward と全体時間へ当てはめると約 0.6〜0.8% の見込みで、全体 1% の線に届かない。
これは推測であり全体の融合実測ではない。新しい融合判定とカーネルの実装は見送り、単体試作を保存する。

### 適用境界と他モデルの数値検収

- `rms-snapshot.log`: 製品の 256 / 128 両 WGSL は実測した試作とバイト一致。
  既存の 256 snapshot を変更せず、128 を追加した。同じキーに異なる本文は渡さない。
- `rms-targeted-test.log`: codegen / RMS / grid-stride / full-write の **95 tests 成功**。
  幅 1,3,17,64,127,128,129,257 で従来版と正本の CPU 許容差を確認する。
  ±0・大きい有限値・非有限値も検査し、NaN は伝播位置、それ以外は u32 一致を求める。
- `rms-mutation-v2.log`: 縮約の上半分を故意に落とす変異で、ビット比較が失敗することを確認した。
  初回は実験 import map の assert 依存が不足し、型検査で止まった。旧ログは残し、製品の条件は変えていない。
- `rms-llm-summary.json`: Deno と Chrome 153.0.8010.36（Vulkan / NVIDIA Ampere、fallback=false）で、
  Qwen f32 / GPTQ I4 を通常 → 128 → 128 → 通常、3 入力 × 64 token 生成した。
  既存 CPU prefill 参照と greedy 列を維持し、**変更経路の全 logits SHA 2,268 比較が一致**した。
  Qwen の RMS は幅 128 が 56 本、幅 1024 が 57 本で、前者だけを変更する。
- MiniCPM5 は RMS 85 本が全て幅 2048 で対象外。追加の **1,134 比較**は非対象経路の維持を確かめたもの。
  最初の集計は MiniCPM5 にも適用対象があると仮定して選択回数の検査に失敗したため、IR の実幅を確認して区別した。
- M2 は未検証。LLM の RMS 比較は数値検収であり、hash を含む時間から速度向上を主張しない。

コミット前の `deno task verify` は **2,849 passed / 743 steps / 0 failed / 5 ignored、24m30s**
（`rms-verify-v2.log`）。最初の実行は追加テストの `finally` 内の throw を lint が拒否し、GPU 実行前に終了した。
後処理後に GPU エラーを報告して元の例外を上書きしない形へ直し、追加 GPU テスト 2 件の再成功後に全体検証をやり直した。
元ログ `rms-verify.log` と `rms-cleanup-test.log` も保存した。検証開始時から 7 個のコード・テストファイルの SHA が同じことを確認した。

## TypeScript の実行費と token-only 出力（2026-09-11）

RTX 3080 Ti、Deno 2.9.6、Chrome 153.0.8010.36（Vulkan / NVIDIA Ampere、fallback=false）での時点調査。
保存先は `outputs/bench/karume/2026-09-11_optimization-next/`。製品の TypeScript 実行経路は変更していない。

### ブラウザの CPU サンプリング

`browser-profile.ts` / `.mjs` で、Qwen3-0.6B の f32 / GPTQ I4、MiniCPM5-2B の f32 を調べた。
各モデルは英語・日本語・WebGPU の 3 入力で 64 token を生成する。読み込み・prefill・最初の 7 decode を外し、
decode 8〜63 の各 56 回、計 168 回を Chrome DevTools Protocol の CPU Profiler で記録した。
サンプリング間隔は 1,000 µs。logits の hash 回収は無効にし、既存 CPU prefill 参照と greedy 列の検査を保った。
容量 128・prefill 16 行の実験グラフであり、CLI の対話処理や表示費は含めない。

以下は renderer のサンプル時間を decode 回数で割ったもの。
`(idle)` には GPU・プロセス間通信・イベントループの待ちなどが含まれるため、GPU の計算時間とは呼ばない。
また、ブラウザ全体や GPU process の CPU 使用率ではない。Profiler 自体の費用もあるので、性能比較は次節の別走行で行う。

| モデル        | JavaScript ms/decode | engine/native ms/decode | idle ms/decode | JavaScript 比率 | idle 比率 |
| ------------- | -------------------: | ----------------------: | -------------: | --------------: | --------: |
| Qwen3 GPTQ I4 |                0.497 |                   0.685 |          8.984 |           4.88% |    88.26% |
| Qwen3 f32     |                0.447 |                   0.698 |         16.573 |           2.52% |    93.54% |
| MiniCPM5 f32  |                0.447 |                   0.609 |         41.861 |           1.04% |    97.47% |

JavaScript の主な観測箇所は `Session.#collectStaged` の出力コピーだった。
個々の dispatch / bind / 記号処理はサンプルが少なく、これだけで細かな順位を断定しない。
`browser-profile-<family>-<0..2>.cpuprofile` が原本、`browser-profile-summary.json` と `host-trials-summary.json` が集計。分類手順は `summarize-profiles.py` に保存した。

ソースも確認し、次の候補は独立した改修に進めなかった。

- `statesOnlySymbols` は入力 shape と記号の集合を調べる。この 3 資産の入力は 2 本で、全中間値の走査ではない。
  構築時 metadata 化は H-10 の既存低優先度判断を維持する。
- `arena.ts` の `toSizeClass` は 4 byte 整列と最小 4 byte であり、2 の冪へ丸めて巨大な余白をコピーする実装ではない。
- `getMappedRange().slice(0)` は unmap 後も利用者へ値を返すために必要な所有コピー。
  単に view へ置換できない。staging の再利用・複数出力の pack は H-9 と同じ寿命・失敗復帰の設計が要る。
- 実行ごとの入力検査・GPU エラー検査・実行リースは維持する。今回のサンプルから削除の根拠は得られていない。

### 全 logits の回収を省く対照実験

現在の Qwen / MiniCPM 実験 CLI は、GPU 内で greedy token を計算するが、IR の出力宣言は `[logits, token]` の 2 本である。次 token だけを使う生成では、この logits 回収量が削減候補になる。

`prepare-token-only.py` で実験用の先頭 shard を別名で作り、**変更は graph.outputs を `[token]` にするだけ**に限った。
その他の graph フィールド、全 tensor payload、残りの shard は同じことを検査した。
証拠は `token-only-projections.json`。既存の系列資産を上書きせず、実験ファイルは保存先の中だけに置いた。
計算ノードは減らしておらず、GPU の logits 計算と argmax は残る。

比較順は全出力 → token-only → token-only → 全出力。3 入力 × 64 token を各方式で実行し、
Profiler と hash 回収は無効にした。各走行の最初の 7 decode を外した 56 回の中央値を取り、往復 2 値の中央値で比べる。
全出力側は既存 CPU prefill logits の `atol=1e-3, rtol=0` を満たし、全方式で既存 greedy 列と **生成した 64 token 全列の一致**を確認した。token-only 側の全 logits 一致を確認したという意味ではない。

decode 最終 run の staging 確保は Qwen の両格納型で **607,748 → 4 bytes**、
MiniCPM5 で **522,244 → 4 bytes**、本数は全て **2 → 1**。
同じ入力間の dispatch 数は同じで、重みと plan backing のバイト数も変わらない。
追加照合は `host-invariants-summary.json` に保存した。

| 環境   | モデル        | 全出力 ms/decode | token-only ms/decode | 速度比（全出力 / token-only） |
| ------ | ------------- | ---------------: | -------------------: | ----------------------------: |
| Chrome | Qwen3 GPTQ I4 |      9.539–9.570 |          8.979–8.988 |                   1.061–1.066 |
| Chrome | Qwen3 f32     |    17.119–17.170 |        16.551–16.667 |                   1.027–1.035 |
| Chrome | MiniCPM5 f32  |    42.344–42.434 |        41.819–41.885 |                   1.013–1.014 |
| Deno   | Qwen3 GPTQ I4 |    19.123–21.214 |        20.872–21.674 |                   0.916–0.979 |
| Deno   | Qwen3 f32     |    27.380–27.501 |        27.472–27.645 |                   0.990–1.001 |
| Deno   | MiniCPM5 f32  |    53.488–53.598 |        53.008–53.504 |                   1.002–1.009 |

範囲は 3 入力の最小・最大であり、信頼区間ではない。原本は `browser-readback-<family>.json` / `deno-readback-<family>.json` と同名ログ、
集計は `summarize-host-trials.py` / `host-trials-summary.json`。
Chrome では全出力に対して約 **1.2〜6.2% の時間短縮**を得たが、Deno の共通改善は確認できなかった。

Deno の Qwen I4 は比較順に沿った時間変動が大きかったため、各 Session に先行 64 token の生成を加えて再測定した。
`deno-readback-warm-qwen3-06b-gptq-i4.json` と `deno-readback-warm-summary.json` では、英語 / 日本語 / WebGPU の速度比は **0.974 / 0.976 / 0.988**。暖機後も改善せず、順序による変動は残った。
この実験は RMS の統合中だったため、import map で両方式を従来 RMS256 に固定した。
原因を Deno 固有の回帰と断定せず、「この条件で利得が再現しない」と結論付ける。追加の GPU 追試はここで打ち切った。

### 採否と残件

H-18 として **将来の LLM 生成用資産での出力宣言を検討する候補**に留める。
現在の `examples/shared/llm-generate.ts` は 2 出力の形を検証するので、token-only 資産はそのままでは使えない。
CLI の検査を外す・loader で IR を黙って書き換える変更は加えていない。
実装する場合は、診断用 logits 出力と生成用 token 出力の用途を recipe / pipeline の契約に明記し、
両環境で複数ターン・中断・エラー・数値を検収する。現行のホスト側で sampling を行う経路には logits が必要で、greedy 限定の比較を一般化しない。
M2・長文・モデル全体の品質は未検証。既存 H-9 / H-10 の単独改修より、この用途の整理を優先候補にする。

## 最適化調査の再開位置（2026-09-11）

序盤の Sol 3 担当による調査は終了し、その後の試作・実測・統合・検収は主担当で行った。
今回の続きで調査エージェントを追加する必要はない。作業ブランチは `codex/review-and-fix`。

| コミット  | 完了した単位                                              |
| --------- | --------------------------------------------------------- |
| `f8bbaf1` | f16 格納 M=1 の GEMV。単体と Deno / Chrome の Qwen を検収 |
| `eaccc9b` | f32 格納 M=1 の GEMV。Qwen / MiniCPM5 を両環境で検収      |
| `cbd0b8c` | QAT INT2 / 固定 SRQ の単体実測と製品化の段階案            |
| `92d219e` | 幅 128 以下の RMS 正規化。Anima 全体と Qwen の数値を検収  |

TypeScript / token-only の記録は、この 4 単位に続く独立した文書コミットとする。
実測の入口は [保存結果の索引](../../outputs/bench/karume/2026-09-11_optimization-next/RESULTS-INDEX.md)。
同ディレクトリの `HANDOFF.md` は開始前のメモなので、最新状態は `STATE.md` の末尾と各 `*-final-status.json` を読む。
再実行する場合は新しい出力先を作り、検証と GPU ベンチを並走させない。

次の推奨は [QAT の製品化段階案](#製品化の段階案判断待ち)の判断と、IR / 固定 SRQ の契約からの段階実装。
PLE の I4 読取、prefill、通常生成全体の一致・品質・メモリを検収するまで、INT2 の単体利得を全体性能として扱わない。
MTP drafter の I8 共有重みはその後の別設計にする。

未完は f16 / f32 / RMS128 の M2 追試、QAT の製品統合・全モデル検収、token-only 出力の正式な recipe / pipeline 契約。
長文容量・広い品質評価・モデル配布・動画対応は従来どおり backlog に残す。今回の CLI と既存モデル資産は変更していない。

この文書コミット前の `deno task verify` は **2,849 passed / 743 steps / 0 failed / 5 ignored、24m26s**
（`host-verify.log`）。追加文書のリンク・数値・保存データとの対応も確認した。
全体検証は終了し、実験用 HTTP / Xvfb / Chrome も停止済み。実行中の GPU ジョブはない。

## M2 の利用者報告と QAT 統合の承認（2026-09-11）

利用者から、M2 で複数モデルの動作を確認したとの報告を受けた。
Anima は約 **400 → 390 秒**（約 2.5% 短縮）、Irodori は約 **35 → 33 秒**（約 5.7% 短縮）。
これは利用者の概算実測で、主担当が M2 を実行した結果ではない。
量子化形式・入力・反復回数・測定ログはこの報告には含まれないため、個々の最適化への寄与を分離せず、
u32 / golden / SHA の自動検収まで完了したという意味にも扱わない。

同じ追加指示で、上記 QAT 製品化の段階案が承認された。
配布・利用者から見える family は **`gemma4-qat`** とし、その内側で **`e2b` / `e4b`** を選ぶ。
公式 mobile 形式は両方存在する（[E2B](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers) /
[E4B](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers)）。
通常 Gemma との共通実行部は再利用し、固定量子化の意味を別 family の明示した契約にする。
決定と実装の分割は [ADR 0097](../decisions/0097-gemma4-qat-integration.md) に記録した。
新しい生データは `outputs/bench/karume/2026-09-11_qat-integration/` に保存する。

AGENTS.md の該当箇所は、無説明の仕様変更を防ぐ意図に合わせて改訂した。
承認された統合範囲は段階ごとに聞き直さず進め、範囲外の変更や前提を覆す問題が出たときに再確認する。

## QAT 統合の INT2 基盤（2026-09-11）

ADR 0097 の承認範囲で、IR の `i2` / safetensors の `I2`、CPU 展開、メモリ見積り、
packed GEMV / GEMM / embedding、shared weight と行分割読込を実装した。
固定 SRQ op・QAT recipe・PLE・別 family の通常生成は後続の単位であり、この時点では未完。
通常 exporter の自動量子化選択は変えず、低レベル writer / reader と固定整数 pack / unpack を追加した。

新しい出力先の `e2b/e4b-download.json` は固定 revision と LFS SHA の照合結果、
`e2b/e4b-census.json` は公式モジュールの論理 shape と実ファイルの照合結果。
E2B の text linear は INT2 61 / INT4 145 / INT8 70 本、E4B は INT2 1 / INT4 258 / INT8 84 本。
PLE は **E2B が INT4、E4B が INT2** なので、後続の host sidecar 読取は両方を扱う。
両モデルとも head と token embedding の全整数列・scale が一致し、調べた重み scale に 0 / 負値 / 非有限値は無かった。

### 製品カーネルの実形状比較

RTX 3080 Ti、Deno。先行実験の固定 E2B up / down / head 重みを用い、丸め前後それぞれを
f32 / INT4 / INT8 / INT2 の GPU 経路で u32 全数比較した。**54 比較すべて一致**。
元の整数・scale を変えず、格納による差だけを見る。

時間は入力 SRQ + linear + 出力 SRQ の合計。ヒータで GPU を起こし、往復順それぞれ 5 標本、
各標本 20 回の GPU timestamp を取り、合計 10 標本の中央値を示す。
これは単体計測であり、モデル全体の生成時間の倍率ではない。

| 形状                 | INT4 GEMV (ms) | INT2 GEMV (ms) | INT4 / INT2 |
| -------------------- | -------------: | -------------: | ----------: |
| up `[12288,1536]`    |      0.0602624 |      0.0479232 |     1.26 倍 |
| down `[1536,12288]`  |      0.2062848 |      0.1386496 |     1.49 倍 |
| head `[262144,1536]` |      0.7132160 |      0.2821888 |     2.53 倍 |

INT2 の M=1 GEMM は同じ順に 0.3359232 / 0.9984768 / 4.7663104 ms。
GEMV の適格条件を満たす単行では GEMV を選ぶ既存の方針を維持する。
行ブロック 4 行の INT2 WGSL は 76,161 文字で、既存の 80,000 文字の目安内。

正本は `int2-product-benchmark-v2.json` / `int2-product-summary.json` と対応する `.ts` / `.log`。
実行には差し替えの無い `product-import-map.json` を使用した。
初回の `int2-product-benchmark.json` は試作 import map を使ったため、正式な数値は v2 に揃える。

保存済み公式 CPU 参照との差は先行実測と件数が完全に一致した。
down/sin と up/near-boundary は符号付きゼロ各 1、up/wide は縮約誤差が SRQ 境界を越す 1 要素。
head/sin と head/wide は各 252,186 要素で、output scale=0 により丸めが無効なため加算順の微小差が残る。
原因の正本は旧出力先の `qat-full-attribution.json`。INT2 が CPU 縮約とのビット同一を達成したとは扱わず、
既存の許容差や golden 条件も変更していない。

### 検証

`int2-targeted-v2.log` は 82 passed / 0 failed。新しい形式検査、GPU の scalar / vec4 GEMM、
GEMV（行ブロック 4 行を含む）、embedding、借用、packed / CPU 展開の行分割、既存 WGSL snapshot を含む。
Python は `int2-exporter-pytest-v2.log` が **3,156 passed / 1 skipped**、
`int2-recipes-pytest.log` が **2,757 passed / 4 skipped**。
未知 dtype の否定テストは `i2` が新しい正規語彙になったため `i1` に変更し、拒否条件を維持した。
コミット前の `deno task verify` は **2,857 passed / 743 steps / 0 failed / 5 ignored、24m30s**
（`int2-verify.log`）。コードは検証した状態のまま、完了後に結果と保存形式の文書表記を更新した。

## QAT 統合の固定 SRQ op（2026-09-11）

INT2 基盤は `ffc532b` に分割コミットした。続く単位では `static_quantize` を IR / CPU 参照 /
GPU / exporter に追加する。具体的な丸め・特殊値・scale の契約は ADR 0097 追記 2。
製品の GPU 実装は除算や乗算を行わず、f32 の正の値のビット順序を使って、境界表と出力表を引く。
CPU 参照は表を使わず、f32 除算と最近接偶数丸めを直接評価する。

### 一致の根拠と回帰データ

出力先は `outputs/bench/karume/2026-09-11_qat-integration/`。
`srq-bit-table-fixture.py` が公式 `transformers.integrations.gemma_quant.apply_srq` の CPU 出力を保存した。
**1,014 scales / 912,068 inputs**（実モデルの値、0、非正規化数、最大有限値、ランダム scale と
丸め境界の両隣、符号付きゼロ、±Inf、符号・payload の異なる NaN）を検査した。

- 独立 TypeScript CPU 参照: **相違 0**（`srq-js-reference.json`）。
- GPU の storage 表試作: **相違 0**（`srq-bit-table-gpu.json` / 実行ログ `-v3.log`）。
- 製品と同じ uniform 表 + grid-stride: **128 / 256 スレッド両方で相違 0**（`srq-uniform-parity.json`）。
  全 scale の入力を 1 workgroup だけで実行し、複数巡回も確認した。
- TypeScript が作る全境界表・出力表も、検証済み Python 表と一致（`srq-params-check.log`）。

この中から特殊 scale 14 種と固定乱数 scale 16 種、計 **24,416 入力**を
`packages/runtime/tests/fixtures/static-quantize-oracle.safetensors` に保持した。
生成の出所は metadata、抽出スクリプトと SHA は `srq-regression-fixture.py` / `.json`。
CPU 参照、Session 経由の GPU、直接 dispatch の grid-stride、汎用 exporter の eager 実装が同じ
期待ビット列を使う。これとは別に、torch.export から `static_quantize_block` の tiny golden を生成し、
既存 golden のバイト列や許容差は変更しない。

### 単体速度の採否

RTX 3080 Ti / Deno、scale=0.09940945357084274、符号を跨ぐ有限入力。
GPU ヒータ後、100 回の dispatch を 1 標本として、往復順各 5 標本を測り、全 10 標本の中央値を比較した。
正本は `srq-uniform-benchmark.json` と対応 `.ts` / `srq-uniform-benchmark-v2.log`。
初回は計測器の timestamp 指定が不正だったため無効とし、修正後の走行だけを集計した。

|    要素数 | 表 128 threads (ms) | 表 256 threads (ms) | 先行の除算 + 境界補正 (ms) |
| --------: | ------------------: | ------------------: | -------------------------: |
|     1,536 |          0.00871936 |          0.00960000 |                 0.00854016 |
|    12,288 |          0.01022464 |          0.01020928 |                 0.00862208 |
|   262,144 |          0.02350592 |          0.02612736 |                 0.01311232 |
| 1,048,576 |          0.07764480 |          0.07031808 |                 0.03056128 |

既定は **128 threads**。decode の活性長で 256 に明確な利得がなく、262,144 要素でも 128 が速い。
表方式には速度コストがある。先行の除算 + 境界補正は通常の有限入力の比較対象であり、
SRQ op が今回受理する特殊 scale / 非正規化数 / 非有限値の全契約を検収した製品経路ではない。
大きな prefill での寄与と融合・表探索の短縮は、全モデルでの帰属後に判断する。

### Python 検証と次段の停止点

SRQ の Python 全体検証は exporter **3,201 passed / 1 skipped**（`srq-exporter-pytest.log`）、
recipes **2,757 passed / 4 skipped**（`srq-recipes-pytest.log`）。

全体 GPU 検証を待つ間、次段の試作を出力ディレクトリ内だけで行った。
固定 writer は INT2 / INT4 / INT8 の重みと scale を、行分割後も全バイト同じまま保持した
（`fixed-writer-proof.json`、`fixed-writer-proof-v3.log`）。製品 writer へは未適用。
公式 CPU の通常生成は E2B / E4B とも英語・日本語各 12 token が完了し、token 列と logits を保存した
（`e2b/e4b-cpu-reference.json` と `*-cpu-en/ja.safetensors`）。

既存 `ProductChunkWrapper` へ QAT text を接続する試作で、同一の英語 prompt について
公式 CPU `generate` の初回 logits と wrapper の最終行を比較した。
E2B は先頭 token=818 が一致し、最大絶対差 **6.67572021484375e-6**。
E4B も先頭 token=818 は一致したが、最大絶対差は **2.6703062057495117** だった
（`e2b-wrapper-proof-v3.log` / `e4b-wrapper-proof.log`）。

E4B の差は未帰属で、共通 wrapper をそのまま採用する検収条件を満たしたとは扱わない。
CPU 同士の比較なので、SRQ の GPU カーネルの問題とも断定できない。
想定外の問題では停止するという利用者の指示に従い、QAT の追加調査・統合を停止した。
比較条件と wrapper 接続の差を中間値まで切り分けることが再開時の判断点。
試作は正式 recipe・配布資産へ適用しておらず、停止状態の詳細は出力先 `STATE.md` に残した。

開始済みだった `deno task verify` は **2,862 passed / 743 steps / 0 failed / 5 ignored、25m53s**
で終了した（`srq-verify.log`、終了コード 0）。製品コードは検証中に変更せず、結果回収後の変更は記録のみ。
SRQ は未コミットで停止した。E4B の原因調査と正式 QAT 統合は未完で、GPU ジョブは終了済み。

利用者から原因調査の継続承認を受けた。検証済み SRQ を独立コミットし、E4B の CPU 比較差から再開する。

## 固定 writer と E4B 比較差の帰属（2026-09-11）

SRQ は `f4752c6` に独立コミットした。次の単位で `FixedQuantizedWeight` と
`write_model` / `publish_model` の `fixed_weights` を実装した。契約は ADR 0097 追記 3。
固定 I2 / I4 / I8 の保存値を f32 へ展開せず、既存の shard 分割と reader 検証を通す。
`fixed-writer-targeted-v4.log` は **26 passed**。全 byte 値、行分割、異なる scale、混成格納、
embedding / linear の共有、通常 f32 定数、曖昧な値指定の拒否、公開失敗時の旧成果物保持を検査した。

初期テストの失敗は新規配線の 2 点だった。公開面の ASCII 昇順と Ruff RUF022 の定数優先が競合したため、
既存テストの順序を保つ局所注記を付けた。また、新しい合成 linear に必須 bias 入力を追加した。
既存テストの期待値・許容差・検証条件は変更していない。

### E4B の差は host RoPE から SRQ 境界へ伝わっていた

`e4b-cpu-attribution.py/json` で、同じ公式固定重みと英語 prompt に対し条件を一つずつ変えた。
保存済み `generate` の初回 logits と、公式 forward の cache あり / なし、text だけの抽出、
PLE 入力の外出し、明示 mask への交換までは **全ビット一致**。
既存 host RoPE の f64 式へ交換した段だけ、最大絶対差 **2.6703062057495117** が再現した。
head の選択を 1 行から 2 行へ変えても、この最大差は同じだった。

`e4b-rope-boundary.py/json` は、wrapper に公式 RoPE 表を注入すると logits が全ビット一致することも確認した。
最初に量子化層の出力が違った箇所は 0 起点 layer 8 の `self_attn.o_proj`。
この層への入力の最大差は **1.9669532775878906e-6** だが、入力 SRQ の相違は 1 要素だけだった。

- 座標 `[0,4,1287]` の丸め前: 公式 **0.596088171005249**、host **0.5960877537727356**。
- 丸め後: 公式 **0.6077758073806763**、host **0.5843998193740845**（差 **0.023375988006591797**）。
- その投影の出力 SRQ では 14 要素に差があり、最大差 **0.04781544208526611**。後続層で差が拡大した。

従来 Gemma の RoPE が f64 計算・最後に f32 格納なのは明示された設計で、未知の重み破損やSRQ opの誤実装ではない。
QAT の不連続な丸めにはこの小差が効く。通常 Gemma の既存数値契約を変更せず、QAT 向けの候補を OUT 内で試す。

候補は `1 / f32(theta ** exponent)` の逆周波数、位置との積を f32 に丸め、三角関数の結果を f32 格納するもの。
公式の CPU 三角関数とは一部に最下位ビット差が残るが、上の英語 prompt の logits は **全ビット一致**になった。
この 1 ケースだけで採用とはせず、E2B / E4B の英語・日本語 12 token 生成まで比較する。
数値列と中間テンソルは同じ OUT の `e4b-rope-boundary-*.safetensors` に保存した。

### 12 token 生成までの CPU 比較

OUT の `qat-f32-rope-reference.py` と `e2b/e4b-f32-rope-reference.json` で、英語・日本語を各 12 token
まで自由な greedy 生成で比較した。**4 ケースすべてで公式 CPU と token 列が一致**した。
logits の全ビット一致は E2B 英語 12/12、日本語 10/12、E4B 英語 10/12、日本語 0/12 step。
各ケースの最大絶対差は順に **0 / 4.10981559753418 / 3.653777599334717 / 5.871218681335449**。
RoPE の最下位ビット差が SRQ 境界へ届く性質は残るため、完全な数値一致や広い品質検収とは扱わない。

対照の従来 f64 式でも、E4B の英語・日本語の各 12 token は公式と一致した
（`qat-f64-rope-reference.py`、`e4b-f64-rope-reference.json`）。logits は両ケースの全 step で差があり、
最大絶対差は英語 **5.152186870574951**、日本語 **5.631443977355957**。
f32 候補は一部の差を除いたが、token 一致に必須と証明したわけではない。製品 RoPE はまだ変更していない。

固定 writer の Python 全体検証は exporter **3,227 passed / 1 skipped**
（`fixed-writer-exporter-pytest.log`）、recipes **2,757 passed / 4 skipped**
（`fixed-writer-recipes-pytest.log`）。

`deno task verify` は **2,862 passed / 743 steps / 0 failed / 5 ignored、26m59s** で終了した
（`fixed-writer-verify.log`、終了コード 0）。実行中の製品コードは固定し、追記した研究記録だけを再整形・確認した。
固定 writer を独立コミットし、次は変換候補の Deno GPU 生成と packed PLE の正式対応へ進む。

## packed PLE の正式対応と初回 GPU 生成（2026-09-11）

固定 writer は `449e7e1` にコミットした。次の単位は ADR 0097 追記 4 の PLE schema 2。
I2 / I4 の論理 shape、格納 byte 数、層別 scale、全量・行読みを共通実装へ追加した。
schema 1 / I8 は維持し、未知版拒否テストの未対応版を 2 から 3 へ進めた。
合成 fixture は実モデルの値を含まず、符号付き整数から Torch の二段 f32 乗算で期待値を作る
（`tools/export-recipes/gemma4/tests/ple_fixture.py`）。packed を展開する TS と同じ式を期待値には使わない。

正式実装の `e2b/e4b-ple-product-proof.json` は公式 probe（E2B 17 token / E4B 11 token）に対し、
全量・行読みとも **全ビット一致**。packed payload の容量は E2B **1,211,105,280 B**、E4B **748,683,264 B**。
各 probe の読み取り総量は全量経路で **1,695,547,012 / 1,247,808,096 B**、行経路で **89,492 / 38,152 B**。
これは重複確認を含む probe の I/O 量であり、速度比較ではない。中断を尊重する読み口と解放も検査した。
既存 I8 と新 packed の対象テストは **26 passed / 29 steps / 0 failed**（`ple-product-targeted-v2.log`）。

### Deno GPU と公式 CPU の生成比較

`qat-gpu-candidate.ts` は正式 runtime と変換候補の固定 packed 重みを使う。容量 128、prefill 物理 32 行、
decode 1 行、head 1 行、linearCompute=f32、greedy 各 12 token。GPU ベンチや他 GPU テストと並走させなかった。
初回の報告で日本語ログだけを見て全ケース一致と述べたが、JSON 全件の集計で誤りと判明したため訂正した。
**CPU 同士の 4 ケース一致と、CPU/GPU 比較の成否を混同しない。**

| GPU 条件                        | E2B 英語           | E2B 日本語    | E4B 英語           | E4B 日本語    |
| ------------------------------- | ------------------ | ------------- | ------------------ | ------------- |
| f32 RoPE / sequential attention | 2 token 目で不一致 | 12 token 一致 | 12 token 一致      | 12 token 一致 |
| f64 RoPE / sequential attention | 2 token 目で不一致 | 12 token 一致 | 4 token 目で不一致 | 12 token 一致 |
| f32 RoPE / parallel attention   | 2 token 目で不一致 | 12 token 一致 | 12 token 一致      | 12 token 一致 |

生データは `e2b/e4b-gpu-candidate-{f32,f64,f32-parallel}.json` と同名 log。
E2B の最初の不一致は公式 CPU の id 16254（score **23.141639709472656**）に対して GPU が id 19656 を選ぶ。
CPU での id 19656 の score は **22.78113555908203**、首位との差は **0.360504150390625**。
E4B f64 の最初の不一致は CPU id 10824 に対し GPU id 496、CPU 側の差は **0.2856578826904297**。
不一致後の logits は異なる token 接頭辞に対する値なので、同条件の誤差として比較しない。

生成後の重み・params アリーナは E2B **828,338,872 B**、E4B **2,306,405,884 B**。
会話用 state はそれぞれ **14,352,392 / 46,792,712 B** で、context.dispose 後には両方 **0 B**。
これは runtime の所有バッファ診断であり、ドライバ全体の VRAM や解放後の OS RSS と同じ量ではない。
通常生成の検収は未完。E2B の最初の相違を中間値まで調べ、正式 family の前に原因と品質上の限界を記録する。

### 中間値と追加検証の準備

E2B の CPU で量子化層の入力・入力丸め後・出力を採取した
（`e2b-gpu-cpu-reference.py`、`e2b-gpu-cpu-reference-v2.log`）。
英語 prompt の自然長 16 行と、GPU と同じ物理 32 行に埋めた入力を比較すると、
有効行の **829 本の保存テンソルが全ビット一致**した（`e2b-cpu-padding-differences.json` は空配列）。
最初の保存試行は Q/K/V などが共有する観測値を safetensors が拒否したため、hook 内で独立コピーして再採取した。
これは観測ファイルの保存方法の修正で、モデルの計算や期待値は変えていない。

GPU 用の `qat-gpu-trace.ts` は固定量子化層の前後を追加出力し、通常出力との比較も保存する。
元の保存 IR に出力一覧だけを加えた検査は GPU 無しで通過した（`qat-gpu-trace-prepare-v2.log`）。
内部 IR の camelCase 表現をそのまま保存 JSON に戻す最初の試行は reader が拒否し、元ヘッダを保持する形へ修正した。
実行前の時点では、GPU と CPU の最初の差の帰属はまだ終わっていない。

固定丸めの数は E2B **487**、E4B **596**、scale=0 は両方 **2**。
恒等な丸めの除去だけに大きな利得を期待せず、`qat-gpu-profile.ts` で GPU 内訳を採る準備をした。
通常の壁時計と、1 dispatch ごとに pass を分ける timestamp 計測は区別する。

PLE の Python 全体検証は exporter **3,227 passed / 1 skipped**（`ple-product-exporter-pytest.log`）、
recipes **2,757 passed / 4 skipped**（`ple-product-recipes-pytest.log`）。

PLE 単位の `deno task verify` は **2,864 passed / 743 steps / 0 failed / 5 ignored、27m45s**
（`ple-product-verify.log`、終了コード 0）。製品コードは走行中に変更せず、結果・作業索引の文書だけを更新した。
fixture の 10 ファイルは保存した生成器から **全 byte 同一**で再生成できた
（`ple-fixture-reproduction.json`）。この単位を独立コミットし、準備済みの GPU 中間値比較へ進む。

## QAT の縮約差の帰属と Chrome 比較（2026-09-11）

以下の生データは `outputs/bench/karume/2026-09-11_qat-integration/`。
PLE は `fc3fd8f` にコミット済み。GPU 中間値の採取を完走し、英語最初の2 step の
通常グラフと追加出力グラフの logits は全ビット一致した。
観測の追加で結果が変わっていないことを確認してから CPU と照合した
（`e2b-gpu-{plain,trace}-f32.json`、`e2b-gpu-trace-comparison.json`）。

最初に固定丸め後の値が異なるのは、0始まり層0の `self_attn.q_proj` 出力の1要素 `[0,4,1055]`。
その linear への入力は SRQ 後に CPU / GPU 全ビット一致していた。
保存した GPU の linear 出力へ CPU の SRQ をかけても、GPU の SRQ 出力と全ビット一致した。
したがって、この地点の差は SRQ の実装差ではなく、**同じ入力の行列縮約差が固定丸め境界をまたぐこと**に帰属できる。

| 値                   |                 CPU |                 GPU |
| -------------------- | ------------------: | ------------------: |
| 丸め前の linear 出力 |  −23.38580894470215 |  −23.38584327697754 |
| scale で除した f32   |  −7.499994277954102 |   −7.50000524520874 |
| SRQ 後               | −21.826770782470703 | −24.944881439208984 |

丸め前の絶対差は **0.000034332275390625**、固定 scale は **3.118110179901123**。
f32 の入力・重みを f64 で内積した対照は **−23.385812002611928**。
CPU / GPU の誤差はそれぞれ約 **+0.00000306 / −0.00003127** だった。
根拠は `e2b-linear-boundary.json` と `e2b-linear-boundary.py`。最初の試行では観測テンソルと
autograd の扱いが衝突し、推論専用に requires_grad を落とした v2 が成功した。
計算順や許容差を変更して結果を合わせる修正は入れていない。

### 8種類の短い生成比較

公式 CPU f32/eager と Deno / Chrome の実 GPU を、greedy 最大16 token で比較した。
ケースは算術・抽出・翻訳・コード・物語・日本語・中国語・会話履歴。
最後のケースは履歴を全量 prefill する比較であり、KV cache を保持する複数ターンの寿命検査ではない。
GPU は f32 RoPE 候補、parallel attention、容量128、物理 chunk を32行単位で入力長に合わせた。

| モデル | CPU と列全体が一致 | Deno と Chrome が一致 | CPU と分岐したケース（0始まり step） |
| ------ | -----------------: | --------------------: | ------------------------------------ |
| E2B    |                6/8 |                   8/8 | 物語13、日本語0                      |
| E4B    |                4/8 |                   8/8 | 物語10、日本語11、中国語9、会話4     |

全件の JSON から集計した `broad-summary.json` が件数の根拠。
CPU は `*-cpu-broad-reference.json`、Deno は `*-gpu-broad-f32-parallel.json`、
Chrome は `*-browser-broad.json`（実 adapter は NVIDIA Ampere、fallback=false）。
英語・日本語各12 token の Chrome 比較も完走し、Deno と同じトークン列だった（`*-browser-short.json`）。
全 logits は有限だった。小さな入力集合なので、これを品質全般の合格とは扱わない。

### 演算別時間

`qat-gpu-profile.ts` / `*-gpu-profile-f32-parallel.json` の内訳を `gpu-profile-summary.json` へ集計した。
2ケースの prefill 各1回、decode 計22回。timestamp 計測は dispatch ごとに pass を分けるため、
通常実行の壁時計やスループットとは異なる。負の timestamp 補正は両モデルとも0件。

| モデル・相  | GPU 合計平均 ms | linear GEMV ms |  SRQ ms |
| ----------- | --------------: | -------------: | ------: |
| E2B prefill |         46.3912 |        32.9663 |  4.6880 |
| E2B decode  |         25.0560 |        14.0455 |  3.6374 |
| E4B prefill |        151.2197 |       122.0135 | 10.0642 |
| E4B decode  |         27.6452 |        17.0304 |  3.1543 |

次の速度候補は linear と SRQ の融合、prefill の GEMV / GEMM 選択。
単独 SRQ が decode の約11〜15%を占めるが、融合でその全量が消えるとは限らない。
まず別 family の通常生成・寿命を統合し、この固定数値契約を保つ A/B で判定する。

### 正式 QAT recipe の全量変換

`gemma4_qat.export` と `dist.py --pipeline gemma4-qat` を追加した（ADR 0097 追記5）。
既存 Gemma の tokenizer・wrapper・PLE 配布検査を使い、固定 payload と scale を保存後に全量照合する。
元ファイルの fingerprint と trace 上限を `reference.json` へ記録し、途中失敗で既存出力を壊さない据え替えに含めた。

`e2b/e4b-recipe-proof/` の全量変換は成功。固定重み数は E2B **276**（I2 61 / I4 145 / I8 70）、
E4B **343**（I2 1 / I4 258 / I8 84）。model / PLE shard 数はそれぞれ **5 / 5**、**10 / 3**。
両モデルで固定重みと PLE の保存後 bytes はすべて元と一致した。ローカル配布形は OUT の
`karume-gemma4-qat/`、組み立て成功ログは `recipe-dist-v3.log`。
初回の graph 門は明示 `f32` 格納を未考慮、次は card の keyword 引数の不一致で失敗し、両方を修正した。
正式 recipe の GPU 8ケースは両モデルとも既存候補とトークン列が全件一致した
（`*-recipe-gpu-broad-f32-parallel.json`、集計 `recipe-generation-parity.json`）。

Python 全体の初回は新 recipe の optional import と据え替え静的検査の登録漏れで2件失敗した。
修正後は exporter **3,227 passed / 1 skipped**、recipes **2,797 passed / 4 skipped**
（`recipe-exporter-pytest-v2.log` / `recipe-recipes-pytest-v2.log`）。
修正中に動いていた初回 Deno verify は終了143で停止したため合格に数えず、製品コードを固定して v2 を実行した。

正式 recipe の `deno task verify` v2 は **2,864 passed / 743 steps / 0 failed / 5 ignored、24m59s**、
終了コード0（`recipe-verify-v2.log`）。実行中は製品コードを固定し、追記した文書は再整形・確認した。
検査済み配布形を新規 `models/karume-gemma4-qat/` に配置済み。次の単位は共通 pipeline と対話 CLI。

## QAT の共通パイプラインと対話 CLI（2026-09-11）

公式 recipe は `022cb09` にコミットした。ADR 0097 追記6に従い、既存 Gemma の本体を非公開の
共通基底へ移し、`Gemma4QatPipeline` / `@karume/models/gemma4-qat` を追加した。
通常 Gemma の会話・MTP・解放の本体を共用する。QAT は専用の構造検査と f32 段丸め RoPE を使う。
QAT の pipeline オプションから MTP を除き、実行時の指定も拒否する。

既存 `demo:gemma4` の対話処理を共通 runner へ移し、`demo:gemma4-qat --model e2b|e4b` を追加した。
model 省略時は manifest の既定。取得元は既定 `models/karume-gemma4-qat/`、明示 `--source` / `--repo` も可。
既定の最大生成は QAT 64、通常 Gemma 256。QAT の実験段階の制約を起動時に表示する。

### Deno / Chrome と会話の寿命

OUT の `family-deno.ts` / `family-browser.ts` は同じ `family-candidate/lifecycle.ts` を使う。
名前に candidate が残るが、実行時の import は正式な公開 `packages/models/gemma4-qat.ts`。
Deno は `denoDirectory`、Chrome はローカル HTTP の全量・区間読みを `localDirectory` へ接続した。
Chrome は NVIDIA Ampere、fallback=false。8ケース・最大16 token、容量128、greedy。

| モデル | chunk      | 既存の単一prefill比較との列一致 | Deno / Chrome の列一致 |
| ------ | ---------- | ------------------------------: | ---------------------: |
| E2B    | 64         |                             8/8 |                    8/8 |
| E2B    | 32（既定） |                             8/8 |                    8/8 |
| E4B    | 64         |                             8/8 |                    8/8 |
| E4B    | 32（既定） |                             7/8 |                    8/8 |

E4B の相違は35 tokenの会話履歴入力で、生成step13（0始まり）から。32行の2chunkに分ける条件と
64行の1chunk条件の比較で、Deno と Chrome はそれぞれ同じ列を出した。縮約と SRQ の感度による差と
**推測**するが、このケースの最初の中間テンソル差は未帰属。chunk 間の全ビット一致は主張しない。
集計は `family-deno-summary.json` / `family-browser-summary.json`、全記録は `*-family-{deno,browser}-*.json`。

全4条件で次を確認した。Deno / Chrome の会話結果は停止理由も含めて一致した。

- 算術2ターン（45 → 4）が EOS で終了し、同じ sequence を1本だけ使う。全履歴の再描画でも2ターン目は4。
- decode中の AbortSignal は指定した理由を返し、doneはaborted。その会話から次の生成が成功する。
- 反復の早期終了はclosedで決着し、次の会話も成功する。reset相当のdispose後も再生成できる。
- 同容量の会話を順に作り直した各runで、state常駐量は E2B **14,352,392 B**、E4B **46,792,712 B** のまま。
  これは会話用stateの量であり、モデル重みや全VRAM量ではない。累積する生存バッファは観測されなかった。

最初の検査は `contextCount` を現存数と誤読して失敗した。この項目は作成した累計本数なので、
定義と実装を確認し、再作成を累計本数・解放を常駐bytesで別々に検査するよう修正した。
この修正で製品の解放処理や期待する常駐量は変更していない。

`e2e_gemma4_qat_test.ts` は公開入口の2モデル・6stepsで成功（`family-product-gpu.log`）。
CPUの形式・RoPE・公開面は15件で成功し、追加で独立Torchの角度丸め期待値を固定した。
CLIはQAT E2B（model省略）/ E4B（明示）/ 通常Gemmaで、2ターン → `/reset` → 1ターン → `/exit` を
標準入力から自動実行し、すべて終了0（`family-cli-results.json`、`gemma4*-*-cli.log`）。手動操作の報告ではない。

### 取得済み資産と末尾 bucket の対照

`family-extra.ts` / `family-extra.json` で、公開 `fromAssets` も両モデルの算術45/EOSを確認した。
E4B の35 token会話履歴は `chunkLength: 32, chunkBuckets: []`（末尾も物理32行）にすると、
chunk64の16 token列と全件一致した。既定bucketでは末尾3 tokenが物理4行となり、
`stateQkParallelEligible`（M<=8）でQKの縮約経路が変わる。対照により小さい末尾bucketの
計画切り替えへの依存を確認した。最初の中間テンソル差の特定はしていないため、単一opの責任とは断定しない。
既定bucketとchunkは変更していない。最初の追加スクリプトはparseManifestへJSON文字列でなくobjectを渡して失敗し、
APIに合わせたv2で検収した（`family-extra-v2.log`）。

### SRQ 後処理融合のマイクロベンチ

`srq-fusion-benchmark.ts` と `srq-fusion-benchmark-steady.ts` は、正式GEMVの縮約本体を保ち、
最後のstoreだけを正式SRQのビット境界表による処理へ置き換える。製品runtimeへの融合はまだ追加していない。
E2Bの実up/down/head重みを使い、I2の同じ整数をI4/I8へ広げた対照も含む。
up/downはM=1/4/32、headはM=1、計21条件。各条件の通常・広い値域・丸め境界近傍の3入力で、
融合前後のu32出力は全件一致した（計63比較、初回・反復増加版とも成功）。

初回は1試料10反復、ABBA順で各候補6試料。時間がばらついたため、両候補を各100反復で交互に3回温め、
1試料100反復・各候補10試料へ増やした。GPU timestampはlinearと後置SRQだけで、前置SRQは両候補に共通で計測外。
個々の実時間・範囲・中央値の正本は `srq-fusion-benchmark{,-steady}.json` と `srq-fusion-summary.json`。

反復増加版のM=1の中央値:

| 形                 | 格納 | 分離 ms | 融合 ms | 速度比（分離/融合） |
| ------------------ | ---- | ------: | ------: | ------------------: |
| up [12288,1536]    | I2   |  0.0425 |  0.0329 |               1.289 |
| up [12288,1536]    | I4   |  0.0580 |  0.0477 |               1.216 |
| up [12288,1536]    | I8   |  0.0674 |  0.0581 |               1.160 |
| down [1536,12288]  | I2   |  0.1154 |  0.1122 |               1.029 |
| down [1536,12288]  | I4   |  0.1908 |  0.1843 |               1.036 |
| down [1536,12288]  | I8   |  0.1875 |  0.1828 |               1.026 |
| head [262144,1536] | I2   |  0.2487 |  0.2430 |               1.024 |

upのM=32はI2で1.003倍、I4で0.988倍、I8で1.061倍。全形状への一律適用の根拠にはしない。
M=1 upの時間短縮は約14〜22%であり、会話で最初に「16〜29%短縮」と書いたのは速度比との取り違えだったため訂正した。
モデル全体の利得・バッファ削減・初回compile費用は未測定。次はdecodeのup/gateに絞った全体比較を優先する。

### 共通化後の host RoPE 計測

`rope-host-bench.ts` は共通化前の `022cb09` と変更後の通常 Gemma、および QAT の RoPE を比較した。
位置1024からの1 / 4 / 32 / 64行を使い、通常 Gemma の変更前後は全u32一致。
100回のwarmup後、前→後→QAT→QAT→後→前を4巡、各試料は約4096位置分を反復した。
1呼び出しの中央値は以下。生データは `rope-host-bench.json`、集計は `rope-host-summary.json`。

| 行数 | 通常・変更前 ms | 通常・変更後 ms |  QAT ms |
| ---: | --------------: | --------------: | ------: |
|    1 |         0.01613 |         0.01592 | 0.01683 |
|    4 |         0.04947 |         0.04847 | 0.05013 |
|   32 |         0.36518 |         0.36015 | 0.36714 |
|   64 |         0.72855 |         0.72151 | 0.73275 |

通常経路の悪化は観測されなかった。約1〜2%の小差は測定の揺れと区別しておらず、高速化の成果とは数えない。
このCPUでQAT decodeのRoPEは約0.017 msであり、まずGPUのlinear/SRQ融合を優先する。

共通パイプライン / CLI の `deno task verify` は **2,869 passed / 760 steps / 0 failed / 5 ignored、25m13s**、
終了コード0（`family-verify.log`）。実行中は実行処理を固定し、追記文書とQAT制約を訂正したCLIコメントを再確認した。
変更文書の相対ファイルリンクは欠損0件。モデル固有 recipe の変更は前コミットで検収済み。

## QAT 統合後の SRQ 融合・探索と INT2 変種（2026-09-11）

共有パイプラインと対話CLIは `7ed64b3` にコミットした。以下は試作で、製品runtimeには追加していない。
この節のスクリプト・生データ・集計・検証ログは `outputs/bench/karume/2026-09-11_qat-integration/`（OUT）に保存した。
RTX 3080 Ti、Deno / Chrome 153（NVIDIA Ampere、fallback=false）。GPUジョブはすべて直列。
実験開始前の採用目安は、出力一致を保ち、全体で約3%以上の安定した利得があり、他モデルで1%以上悪化しないこと。

### SRQ 融合の全モデル比較

`srq-full-candidate/` はdecodeのup/gateだけ、`srq-broad-candidate/` はGEMV適格な全固定量子化linearを対象とする。
どちらもM=1・f32計算に限定し、privateなlinear出力と後置SRQを1dispatchへまとめる。
固定weight scaleは既存の常駐バッファを借り、SRQの境界表はGEMVのuniformへ同居させた。
元のK縮約とbias順を変えず、出力storeに整数境界表によるSRQを適用する。

`srq-full-parity.ts` / `srq-broad-parity.ts` は、正式recipeの8入力・最大16tokenについて
各stepのlogitsとhiddenをSHA-256で比較した。通常版と各候補で、E2B **196** / E4B **208**、
計**404出力ハッシュ**がそれぞれ全件一致。decodeの融合数は限定版 **70 / 84**、広い版 **275 / 342**。
prefillの融合数は0。集計は `srq-full-summary.json` / `srq-broad-parity-summary.json`。

全体ベンチは通常版と候補を同じGPU上で交互に実行する。モデルはそれぞれ別deviceへ読み込み、実行を並走させない。
まず8ケース×最大16tokenを両候補で比較し、次に物語・日本語・コードを最大64tokenで
通常→候補→候補→通常の順に3巡する。各候補・各課題6試料。greedy、容量128、既定chunk32、
PLEは行読み（常駐予算0）。診断のtimestamp計測は有効にせず、通常runのwallとdecode相を記録した。
以下はwall中央値の速度比（通常/候補）。初回ロードは順番とcacheの影響を分離していないので、速度比較へ使わない。

| 候補         | 環境   | モデル |   物語 | 日本語 | コード |
| ------------ | ------ | ------ | -----: | -----: | -----: |
| up/gate融合  | Deno   | E2B    | 1.0092 | 1.0048 | 1.0034 |
| up/gate融合  | Deno   | E4B    | 1.0106 | 1.0034 | 1.0017 |
| 全linear融合 | Deno   | E2B    | 1.0179 | 1.0198 | 1.0176 |
| 全linear融合 | Deno   | E4B    | 1.0155 | 1.0031 | 1.0225 |
| 全linear融合 | Chrome | E2B    | 1.0266 | 1.0208 | 1.0384 |
| 全linear融合 | Chrome | E4B    | 1.0001 | 1.0076 | 1.0400 |

Chromeの一部課題は3%を超えたが、同じモデルの他課題で安定した利得にはならない。
Denoの全体では最大でも約2.2%。新しい融合ルール・常駐weight用operand・uniform構成を製品へ追加する根拠としては不足し、
今回は両融合を見送る。生成列は全反復で一致。広い融合の最大64token比較はDeno/Chrome間も12条件すべて一致した
（`srq-broad-cross-backend.json`）。M2での速度・同値性は未検証。

生データ: `*-srq-full-bench.json`、`*-srq-broad-bench.json`、`*-srq-broad-browser.json`。
集計: `srq-full-summary.json`、`srq-broad-bench-summary.json`、`srq-broad-browser-summary.json`。

### INT2 GEMV の担当列数と先読み

`int2-gemv-sweep.ts` はE2Bの実up/down/headで、現行c32u4と列数16/32/64/128×先読み2/8/16の計13変種を比較した。
3形×13=39条件で、通常・広い値域・境界付近の3入力を使用。SRQ前の生linear出力はすべてu32一致。
同じ変種を100回温め、100反復/試料、順番を往復し各6試料。GPU timestampの中央値を比較した。

| 形                 | 現行c32u4 ms | 最良変種 | 最良 ms | 速度比 |
| ------------------ | -----------: | -------- | ------: | -----: |
| up [12288,1536]    |      0.03094 | c32u4    | 0.03094 |  1.000 |
| down [1536,12288]  |      0.11149 | c64u8    | 0.10639 |  1.048 |
| head [262144,1536] |      0.24333 | c32u8    | 0.24287 |  1.002 |

upは現行が最良、headはほぼ同等。downの小差を全体へ外挿すると寄与は限定的で、形状別の選択を増やす根拠にしない。
ばらつきもあり、全モデル実装やM2の最適値の主張へは進めない。生データ `int2-gemv-sweep.json`、
全変種の範囲・初回compile時間を含む集計 `int2-gemv-sweep-summary.json`。既定変種は変更していない。

### SRQ 境界探索の別案

`srq-search-candidate/` は融合を追加せず、SRQ単体だけを変更する。浮動小数の商から探索の開始位置を概算し、
整数境界表で上下に移動して最終levelを確定する。概算の精度を数値契約にせず、元と同じ順序付き境界の比較で決める。
非正規scaleは従来の整数二分探索を使う。ゼロscaleの恒等写像、NaN・Inf・符号・飽和・出力値の表は維持する。

既存の独立Torch oracle **24,416入力**を、直接dispatchとSession（反復含む）で全ビット照合し、4テスト成功。
さらに両モデルの計**404出力ハッシュ**も通常版と全件一致した
（`srq-search-test.log` / `srq-search-parity-summary.json`）。
単体はup/down/headの出力要素数・実scaleに、量子化levelの振幅8/130を使う6条件。
各候補100反復×4試料の速度比は**1.003〜1.081**、通常版の絶対時間は約0.008〜0.010 ms。
小さい演算なので、単体比率だけで全体の利得とは扱わない（`srq-search-micro{,-summary}.json`）。

全体ベンチは融合と同じ8ケースの対照・3課題のABBAを使った。候補名がログの`fused`に残るが、
この試作はSRQ単体の探索変更だけで、linear融合はない。wall中央値の速度比:

| 環境   | モデル |   物語 | 日本語 | コード |
| ------ | ------ | -----: | -----: | -----: |
| Deno   | E2B    | 1.0075 | 1.0058 | 1.0046 |
| Deno   | E4B    | 1.0035 | 1.0083 | 1.0054 |
| Chrome | E2B    | 1.0019 | 0.9999 | 1.0052 |
| Chrome | E4B    | 1.0119 | 1.0029 | 0.9754 |

Denoは全課題1%未満、ChromeのE4Bコードは約2.5%遅い。採用条件を満たさず、この探索変更も見送る。
正当性検査の成功だけを高速化の根拠にはしない。生データ `*-srq-search-{bench,browser}.json`、
全範囲・decode時間の集計 `srq-search-summary.json`。

### compute pass の切り方を揃えた対照

初回と反復増加版の融合マイクロベンチは、各dispatchを別compute passに置いていた。
通常推論は `SubmitScheduler.#encodePlainChunk` が複数dispatchを同じpassにまとめ、copyが来たときだけ区切る。
診断のtimestamp計測は別passに分けるため、単体の測定方法による費用を全体利得へ外挿できない。

`srq-fusion-single-pass.ts` は同じ入力・WGSL・反復数を保ち、比較するdispatch列を1つのpassへ置いた。
21条件×3入力のu32比較はすべて成功。各候補100反復×10試料のM=1の速度比は、
up I2/I4/I8が **1.084 / 1.061 / 1.041**、downが **0.930 / 1.004 / 1.011**。
別pass時のup **1.289 / 1.216 / 1.160**から差が縮まり、down I2は順位も逆転した。
境界の違いが単体の結論に影響することを確認した。唯一の原因とは断定せず、採否は通常モードの全体wallで判断する。
生データ `srq-fusion-single-pass.json`、全21形・範囲の集計 `srq-fusion-single-pass-summary.json`。

今回の全体A/Bは各52生成×10実行=**520生成**。広い融合と探索変更の最大64token比較は、
Deno/Chrome間の計24条件すべてで列が一致した（`optimization-final-audit.json`）。
製品への追加最適化は見送り、既定カーネル・数値契約・APIを変更しない。
次はM2のQAT検収と、既存H-18の生成用出力・転送削減の設計を優先する。品質・長文・M2は未検収のまま区別する。

### 転送削減の契約確認と CPU 抽選の追加観測

全体検証中にGPUを使わず、H-18の次段をソースで確認した。現行 `Session.run` は読み戻す出力を選ぶ引数を持たず、
`#stageOutputs` はgraph.outputs全件を読み戻す。`GenerationProgram` はlogitsとhiddenを必須としている。
通常Gemma/QATの温度0でもrepetition penaltyとlogit biasは有効で、NaNや最大値が非有限の入力は拒否する。
GPUのargmaxだけへ置き換えるなら、その適用条件と異常値の拒否を明示する必要がある。今回は契約を変更していない。

`sampler-host-bench.ts` は保存済みの公式CPU logits（E2B/E4B、物語の先頭と最終行）を使い、
`createSampler().next()`を64回温め、128回×5試料で計測した。これはこのLinux CPU上の観測で、M2の値ではない。
GPUの全体検証と並行したCPU計測なので、絶対時間は処理の規模を知る材料とし、全体の高速化率には使わない。
既定のtop-k併用とtop-p単独では費用が大きく違う（`sampler-host-bench.json` / `sampler-host-summary.json`）。
各行の中央値の範囲はgreedy **0.2004〜0.2097 ms**、QAT既定 **0.2695〜0.3464 ms**、
top-p単独 **14.1157〜14.1544 ms**。後続の分布計測は別API・別条件であり、この絶対値とは直接比較しない。

基数幅の候補もQAT実logits4行と合成1行で再測定した。`samplerDistribution`の候補idと確率のu64列はすべて一致。
8/10/11/12/16bitの比較で16bitは約1.19〜1.21倍だったが、過去の **H-17** と同じ案であることの確認が後になった。
小語彙の回帰とhistogram増という既存の不採用理由を覆す測定ではないため、再実装やサイズ切替は進めない。
この追加は新規の高速化成果に数えず、QATでの再観測として残す
（`sampler-radix-bench.json` / `sampler-radix-summary.json`、各16反復×6試料・順序反転）。

### この調査単位の最終検証

調査記録のコミット前に `deno task verify` を新規実行し、fmt・lint・型検査と実GPUテストを完走した。
**2,869 passed / 760 steps / 0 failed / 5 ignored、25分2秒、終了コード0**。
ログは OUT の `research-verify.log`。この走行中に他のGPUジョブは実行していない。
製品コードは `7ed64b3` のまま。結果の追記後の整形検査775ファイル、相対ファイルリンク303件、差分検査も成功した。

## 停止条件の解釈訂正（2026-09-11）

利用者から「想定外なら停止」はデータ損失・誤操作などのインシデントを指すと明確化された。
本記録の以前の性能低下・Qwen参照差・QAT E4B参照差での停止は、当時の対応履歴として残すが、
今後の停止条件には使わない。数値差や性能回帰を見つけた場合も、ログと不変条件を保ち、
原因調査・改善・候補の不採用判断を続ける。期待値や許容差を弱めて合格させる許可ではない。

AGENTS.md、ADR 0097、再開用引き継ぎの記述をこの意図へ揃えた。
今回の公開カーネル調査では、利用者から序盤の調査エージェント使用も改めて許可された。
棚卸し後の実験・修正・検収は主担当で行う。

この文書変更のコミット前に `deno task verify` を実行し、
**2,869 passed / 760 steps / 0 failed / 5 ignored、24分59秒、終了コード0**。
ログは `outputs/bench/karume/2026-09-11_fable-followup/policy-verify.log`。
同時に別のGPUジョブは実行していない。製品コードは変更していない。

## 公開カーネル調査と非同期コンパイル（2026-09-11）

以下はRTX 3080 Ti / Deno 2.9.6 / Chrome 153.0.8010.36の時点実測。
今回の OUT は `outputs/bench/karume/2026-09-11_fable-followup/`。
利用者のM2観測はDenoのdemo表示で、通常E2Bが初回6.8 / 2回目16.4 tok/s、
QAT E2Bが初回4〜5 / 2回目16〜17 tok/s。prompt・生成長を固定した比較ではなく、
以下のRTX実測とも混ぜない。利用者は初回と定常生成の両方の改善を希望している。

### 公開実装との差

[公開Spaceの固定版](https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels/tree/158f16ae0f672943ca304d59c47c8e3a264e399e)を
`~/workspace/reference/gemma-4-webgpu-kernels` で確認した。
GoogleのE2B mobile QATを読む独立したbundleで、Transformers.js / ONNX Runtimeを呼び出す形ではない。
`index.html` はload後にwarmupし、速度を最初のトークン以降で算出する。karumeのCLI表示はターン全体である。
GPUでの最大値選択、小さい読み戻し、次tokenのGPU保持、先行submit、QKV / MLP / norm等の融合が主な構成差だった。
縮約順・SRQ・GELUの実装も違うため、公開デモを数値の正解としては使わない。
root / README / bundleにコードのライセンス指定は見当たらず、製品へそのソースを複製していない。

関連する上流の変更として、[ORT #28501](https://github.com/microsoft/onnxruntime/pull/28501)の共有KV対応、
[ORT #28280](https://github.com/microsoft/onnxruntime/pull/28280)のQKV / MLP融合、
[ORT #29557](https://github.com/microsoft/onnxruntime/pull/29557)の非同期コンパイルを確認した。
公開SpaceのbundleそのものがORT / Transformers.jsへマージされたことは確認していない。
karumeにも共有KVと導出済み計画はあるが、パイプラインの生成待ちは直列だった。

### 初回と定常生成を分けた計測

`cold-warm.ts` / `cold-browser.ts` を使用。通常E2B i4とQAT E2Bをcapacity128 / chunk32 / PLE常駐0、
同じ入力token、温度0、最大64 tokenで比較した。英語短文・日本語・コードの3件を各3回、
同じpipelineの新しいsequenceで生成した。実際の複数ターン会話ではなく、初回とキャッシュ済み実行の比較である。
各runの壁時計・出力バイト・診断、シェーダー生成の時間と全文、最初のtokenまでの時間をJSONへ保存した。
計測フックのCPU費用も含む。token数に停止tokenを含む全体速度と、配送した最初のtoken以降の速度は別の欄にした。

Deno基準は `normal-cold-0.json` / `qat-cold-0.json`。初回応答は通常331 ms / QAT507 ms、
シェーダーモジュール生成は113 / 249 ms、同期pipeline生成は12 / 21 ms、新規キーは32 / 43本だった。
64 token生成の暖機後は双方約37 tok/s。QATの軽量化だけでは定常速度差が出ない傾向が再現した。
Chrome QAT基準の暖機後は約49 tok/s。全3件のtoken列はDenoと一致した。
Denoの内部待ちについては、本記録の「ホスト待ちと既存融合の追加測定」の帰属を参照する。

### 非同期コンパイルの比較と採用範囲

`async-import-map.json` で自分のruntimeの隔離コピーへ差し替えた。
WGSLとGPUでの計算順序を変えず、非同期コンパイルとレシピの並列準備を組み合わせた。
初回応答は次のとおり。Chrome QATは基準→候補→基準→候補と再測定した。

| 実行環境 / モデル       |     基準 |     候補 | 生データ                                                   |
| ----------------------- | -------: | -------: | ---------------------------------------------------------- |
| Deno / 通常E2B          | 330.9 ms | 328.2 ms | `normal-cold-0.json` / `normal-async-0.json`               |
| Deno / QAT E2B          | 506.9 ms | 512.3 ms | `qat-cold-0.json` / `qat-async-0.json`                     |
| Chrome / 通常E2B        | 711.5 ms | 550.1 ms | `chrome-normal-base-0.json` / `chrome-normal-async-0.json` |
| Chrome / QAT E2B・1回目 | 879.3 ms | 499.5 ms | `chrome-qat-base-0.json` / `chrome-qat-async-0.json`       |
| Chrome / QAT E2B・2回目 | 803.4 ms | 508.1 ms | `chrome-qat-base-1.json` / `chrome-qat-async-1.json`       |

レシピ並列準備だけの対照は967.5 ms（`chrome-qat-build-0.json`）で改善しなかった。
基準と候補の全9生成、またChrome QATの追加9生成でtoken列が一致した。
Denoの定常速度はほぼ横ばい。Chromeの定常値は揺れがあり、通常版の短い日本語生成では1走行で約16%遅かった。
初回の改善を定常速度の改善とは扱わない。M2の初回差をこの結果だけで解決済みとも呼ばない。

採用する変更は共通runtimeの初回準備だけ。公開API・WGSL・キー・GPUのdispatch順序は維持する。
失敗したコンパイルを待ち切る、原因を保持する、逆順完了でもレシピ順を維持するテストを追加した。
シェーダー診断の補完は、既存L-7の調査で必要とされていた条件を維持する。
詳細の決定は [ADR 0042](../decisions/0042-prepared-execution-plan.md#非同期コンパイル2026-09-11)。

比較5組・合計45生成のtoken列と、キーごとのWGSL全文が一致した（`async-summary.json`）。
製品へ反映した後のキャッシュ・依存順・error scope・計画キャッシュの重点検証は
**33 passed / 0 failed、2秒**（`async-product-focused-v2.log`）。

初回の全体検証は **2,875 passed / 1 failed / 5 ignored、24分49秒**。
失敗は `gpu_resident_batch_test.ts` の故障注入が旧同期APIへ残り、注入回数が0だったため。
当該ファイル単独でも **24 passed / 1 failed** と再現した。VRAM圧による失敗とは扱わない。
注入先を非同期APIへ合わせ、参照数・破棄拒否・出力ビット・完了後の解放の期待値は維持した。
修正後は既存の重点検証に常駐テンソルの検証を加え、**58 passed / 0 failed、7秒**。
ログは `async-product-verify.log` / `async-resident-isolate.log` / `async-product-focused-v3.log`。

修正後の `deno task verify` は **2,876 passed / 760 steps / 0 failed / 5 ignored、26分30秒、終了コード0**。
ログは `async-product-verify-v2.log`。他のGPUジョブは並走させていない。
この走行の一部では公式CPU参照の生成も行ったため、検証所要時間を性能比較には用いない。

## INT2 行ブロックの初回準備短縮（2026-09-11）

RTX 3080 Ti / Deno 2.9.6 / Chrome 153での時点実測。OUTと入力・生成条件は直前の非同期コンパイル節と同じ。
本節の基準は非同期コンパイル反映後。利用者のM2の値とは区別する。

### 候補を絞った理由

INT2は16 B語が64要素を表し、従来は語の先読みと行数ぶん積和を展開していた。
`helper-i2.ts` は計算順を保ったWGSL関数化、`small-workgroup-i2.ts` はcols4/8/16/32/64、
`select-i2.ts` は4値の逆量子化を縮約前に計算する候補を比較した。
実際のdown / up / head重み、3入力、同じcompute pass内の100反復×8試料、試料ごとの順序反転で測定した。
生データは同名JSON、集計は `exact-micro-summary.json`。全候補で基準とのraw u32は一致した。

cols縮小はdownで最大1.009倍、up/headで大きく遅くなり採らない。
関数化したM=1はdownで0.907倍、up/headで約0.997倍。逆量子化の4値選択はdownで1.086倍、
upで0.975倍、headで1.022倍だった。4値選択の全体効果は未検収で、製品には入れていない。

M=1と行ブロックを共通化した全体候補 `helper-import-map.json` は、Denoの初回応答を
512→388 / 513→397 msに短縮したが、暖機後の中央値は両走行とも基準の約0.993倍。
Chrome初回は544→743 msへ遅くなった（`qat-helper-0/1.json`、`qat-helper-base-1.json`、
`chrome-qat-helper-base-0.json` / `chrome-qat-helper-0.json`）。この全域適用は採らない。

### 行ブロックだけを変更した対照

`helper-rows-import-map.json` はM=1のWGSLを従来のまま残し、行ブロックだけ関数化した。
既定r4のWGSLは **76,161→9,169文字**。r1/r2も7,828 / 8,275文字となる。
幾何や行数の選択を変えず、生成テキストと初回の解析量を減らす。

| 実行環境      | 基準の初回応答 | 候補の初回応答 | 生データ                                                               |
| ------------- | -------------: | -------------: | ---------------------------------------------------------------------- |
| Deno          |       513.0 ms |       405.3 ms | `qat-helper-base-1.json` / `qat-helper-rows-0.json`                    |
| Chrome・1組目 |       543.5 ms |       463.6 ms | `chrome-qat-helper-base-0.json` / `chrome-qat-helper-rows-0.json`      |
| Chrome・逆順  |       493.5 ms |       414.7 ms | `chrome-qat-helper-rows-base-1.json` / `chrome-qat-helper-rows-1.json` |

Chromeは候補→基準の順でも初回短縮を再現した。全3組・27生成のtoken列が一致。
暖機後の全体速度比はDeno **0.996〜1.006**、Chrome **0.996〜1.067**で、短文の揺れを含む。
M=1のWGSLは全文不変なので、この値を生成カーネルの高速化とは呼ばない。
集計は `new-model-summary.json`。公開API、数値契約、保存形式を維持し、
変更する行ブロックだけキーの版を上げる（[ADR 0082](../decisions/0082-linear-gemv-decode.md#追記-82026-09-11-int2-行ブロックのシェーダーを縮小するk-30)）。

### 正しさと適用範囲

`helper-real-bits.ts` / `helper-rows-real-bits.ts` は実重み3形×M=1/2/4/8/16/32×3入力、
**54条件・52,157,952要素**を基準と比較し、raw u32の差は0。
製品へ整理した生成器は `check-i2-rows-product.ts` で実測候補とのWGSL全文一致を確認した。
M=1と他格納型のsnapshotは更新せず、変更対象のINT2 r4のみ更新した。

製品の重点検証は **90 passed / 0 failed、2秒**（`i2-rows-product-focused.log`）。
K=12,288や先読みの残り、r1/r2/r4、最終タイルの端を、非2冪scale・biasと幅の異なる入力で検査し、
既存M=1の出力とビット一致した。行ブロックの既定全高さに10 KB未満の生成量の門も加えた。
M2、E4B全体、長文prefillは未計測。全体検証はこの重点検証とは別に行う。

整形後の製品コードでも、凍結した旧カーネルを対照として同じ54条件・52,157,952要素のu32一致を再確認した
（`i2-rows-product-real-bits.ts` / `.json` / `.log`）。この対照は現在の製品ファイルを参照せず、
保存済みの `async-candidate/runtime` のカーネルを直接読む。

コミット前の `deno task verify` は **2,878 passed / 760 steps / 0 failed / 5 ignored、24分38秒、終了コード0**。
ログは `i2-rows-product-verify-v2.log`。最初の起動はリンク修正後の表整形で停止したため、
整形を揃えて全体を再実行した（最初のログは `i2-rows-product-verify.log`）。
この全体検証中、CPU専用と誤認した別の `estimate_test.ts` に実GPUの検証が1件含まれ、約199 ms重なった。
性能ベンチは並走させていない。以後はテスト名だけでCPU専用と判断しない。

## PLE の行内並行読み（2026-09-11）

この節は2026-09-11時点の実測と採用判断。生データは
`outputs/bench/karume/2026-09-11_fable-followup/`、集計は`new-model-summary.json`。
PLE（per-layer embeddings、層ごとの埋め込み）の行読みには、値と復元用scaleの2区間がある。
既存実装はこの2回を直列に待つ。1トークン生成では、同時読み込み上限16に対して1本しか使っていなかった。

一意行数が8以下なら値とscaleを並行して読み、それより多ければ従来の16workerと行内直列を保つ。
数値演算と配布バイトは変えず、公開APIも追加しない。
設計と失敗時の意味は[ADR 0085](../decisions/0085-ple-host-gather.md#追記2026-09-11--小さい-gather-の値と-scale-を並行して読む)へ記録した。

### モデル全体の対照

RTX 3080 Ti、Deno 2.9.6とChrome 153。容量128・chunk32・PLE常駐0・温度0・最大64tokenで、
3入力を同じpipeline上で各3回生成する。毎回新しいsequenceを作るため、実際の多ターン会話の計測ではない。
以下はrepeat 1/2の6生成に対する「基準の所要時間÷候補の所要時間」。1より大きいほど速い。
Chromeの取得元はローカルHTTP Range、Denoはローカルファイル。CacheStorage経由の速度は未計測。

| 環境・モデル            | 速度比の最小 / 中央 / 最大 | 生データ（基準 / 候補、拡張子は.json）             |
| ----------------------- | -------------------------- | -------------------------------------------------- |
| Deno・通常E2B           | 0.992 / 1.001 / 1.015      | `normal-ple-base-0` / `normal-ple-0`               |
| Deno・QAT E2B           | 0.999 / 1.006 / 1.008      | `qat-ple-base-0` / `qat-ple-0`                     |
| Chrome・通常E2B         | 0.993 / 1.017 / 1.042      | `chrome-normal-ple-base-0` / `chrome-normal-ple-0` |
| Chrome・QAT E2B、順方向 | 1.023 / 1.048 / 1.051      | `chrome-qat-ple-base-0` / `chrome-qat-ple-0`       |
| Chrome・QAT E2B、逆方向 | 1.017 / 1.020 / 1.029      | `chrome-qat-ple-base-1` / `chrome-qat-ple-1`       |

各組9生成のtoken列が一致した。QAT Chromeの基準→候補→候補→基準でも利益が残り、
この取得元の定常待ちを減らす小さな改善として採用する。Denoに確かな高速化が出たとは主張しない。
通常版も同じloaderを使うため適用されるが、M2・E4B全体・長文prefillの速度は未検収。

### 正しさの検証

同時読み込み数を1/8/9/16行の境界で観測し、順に2/16/9/16本となることを確認する。
片方だけ、または両方の読み込みを拒否する3条件では、相手の未完了読み込みを待ち切ること、
元の例外を保持すること、失敗行を完了数へ含めないこと、その後に同じ行を引けることを検査する。
既存の全量・行読みのu32比較、packed INT2/INT4の復元、中断signalの検証も維持する。

製品版の重点検証は **33 passed / 29 steps / 0 failed、139 ms**
（`ple-product-focused-v2.log`）。最初の検証は追加テストの未設定importで型検査に失敗したため、
依存を増やさず既存の`Deno.test`形式へ揃えて再実行した。検証条件の緩和はない。

コミット前の `deno task verify` は **2,885 passed / 760 steps / 0 failed / 5 ignored、24分38秒、終了コード0**
（`ple-product-verify.log`）。GPUベンチや別のGPUテストは並走させずに完了した。

## INT4・INT8 行ブロックとドライバキャッシュの分離（2026-09-11）

この節はRTX 3080 Ti / Deno 2.9.6 / Chrome 153での時点実測。生データのOUTは
`outputs/bench/karume/2026-09-11_fable-followup/`。通常版E2BとQAT E2Bを別々に比較する。
INT2だけの関数化を基準とし、INT4・INT8も行ブロック内の語を関数化した。
最大高さの生成文字数はINT4 r8が72,592→7,835、INT8 r16が70,318→10,212。
M=1の全格納型とINT2行ブロックは生成WGSL・キーとも不変。

### 最初の比較で生じた逆転

`all-rows-model-summary.json` は3入力×各3生成の基準・候補を2組比較した集計。
各組9生成のtoken列は一致した。Deno通常版の初回は333→185 / 341→184 msに短縮したが、
QATは401→799 / 414→263 ms、Chrome通常版は467→620 / 481→338 ms、
Chrome QATは428→645 / 431→311 msとなった。
候補の最初の起動だけ遅い傾向から、ドライバの永続キャッシュを切り分けた。

暖機後6生成の速度比中央値はDeno通常版0.999 / 1.004、QAT0.994 / 0.998、
Chrome通常版0.989 / 1.015、QAT0.991 / 0.995。
M=1は変更しておらず、この揺れから定常速度の改善を主張しない。

### 空キャッシュと再利用の対照

`run-cold-cache.py` と `run-cold-cache-reverse.py` は、環境・モデル・候補ごとに新しい空の
専用ディレクトリを作り、子プロセスだけに `__GL_SHADER_DISK_CACHE_PATH` を設定する。
1回目のプロセス終了後、同じディレクトリを使う2回目のプロセスを起動する。
既存キャッシュを消さず、設定も永続変更しない。逆順では環境・モデル・候補の順序を反転し、別の空ディレクトリを使った。
各条件で2ファイルの作成と次プロセスでの同じ容量を記録した（`*-cache-files.json`）。
[NVIDIAの環境変数の説明](https://http.download.nvidia.com/XFree86/Linux-x86_64/555.58/README/openglenvvariables.html)は
OpenGLについての文書であり、Vulkanの仕様根拠とはしない。ここでは実際のファイル生成と時間差を観測した。

容量128・chunk32・PLE常駐0・温度0で短い物語を2token生成した際の、最初のtokenまでの時間を測る。
以下の「再利用」も新しいプロセスの初回応答であり、同一会話の2ターン目ではない。
基準はINT2行ブロックのみ関数化済み、候補はINT4・INT8も適用済み。

| 環境・モデル  | 順序   | 空キャッシュ 基準→候補 | 再利用 基準→候補 |
| ------------- | ------ | ---------------------: | ---------------: |
| Deno 通常版   | 順方向 |         985.9→863.7 ms |   333.2→186.5 ms |
| Deno 通常版   | 逆方向 |         977.6→857.6 ms |   336.4→182.2 ms |
| Deno QAT      | 順方向 |       1643.8→1525.5 ms |   405.3→260.3 ms |
| Deno QAT      | 逆方向 |       1650.6→1526.2 ms |   407.1→269.6 ms |
| Chrome 通常版 | 順方向 |         797.7→704.3 ms |   402.9→240.0 ms |
| Chrome 通常版 | 逆方向 |         816.7→706.6 ms |   364.1→343.8 ms |
| Chrome QAT    | 順方向 |         856.7→788.0 ms |   437.1→303.5 ms |
| Chrome QAT    | 逆方向 |         837.0→869.9 ms |   471.9→357.7 ms |

全条件で同じ環境・モデルのtoken列が一致した。集計は `cold-cache-summary.json` /
`cold-cache-reverse-summary.json`、個票は `{deno|chrome}-{normal|qat}-{cache|cache2}-{i2rows|allrows}-{0|1}.json`。
INT2の関数化前を示す `async` 対照もQATで各2回含む（合計40プロセス）。
推測: 前の比較は既にコンパイル済みの基準と、候補の初回ドライバコンパイルが混在し、逆転が大きく出ていた。
ただしChrome QATの空キャッシュ逆順は約4%悪化しており、全条件での短縮とはしない。

Deno通常版のシェーダー生成APIの同期部分は約113→24 ms、QATでは約159→57 msへ短縮した。
非同期コンパイルのpromise時間には重複とJSの待ちも含むため、各promiseの時間を足してCPU費用と扱わない。
M2・E4B全体・長文prefillは未計測。

### 統合と検証

最初の候補の既存GPUテストは6件成功。整理した生成器は、M=1のcols16/32×unroll1/4/8で
旧生成器との全文・キー一致、全既定行高さで測定候補との全文一致を確認した
（`all-rows-minimal-source-check.ts` / `.log`）。
INT4・INT8のsnapshot4本だけを更新し、M=1とINT2は更新しない。
K=4096・M=9・N=4096の長い縮約と部分タイルを両格納型のGPUテストへ追加し、
通常GEMMの先頭行とのu32一致・独立CPU参照・実際の選択キーで検証する。
決定は[ADR 0082 追記9](../decisions/0082-linear-gemv-decode.md#追記-92026-09-11-int4int8-行ブロックにも関数化を適用する)。

追加した長いKの最初の検証は **90 passed / 2 failed、8秒**。
通常GEMMとの全ビット比較は通り、f64で総和する汎用CPU参照との比較だけが失敗した。
凍結した変更前カーネルでも同じ2件・同じ最大誤差で再現した
（`all-rows-product-focused.log` / `all-rows-old-long-isolate-v2.log`）。
INT4はmaxAbs 0.00042724609375、INT8は0.002838134765625。
既存のGEMM許容差はK=72までのコーパスに由来し、K=4096のこの入力への適用は不適切だった。
独立CPU参照を乗算・加算ごとにf32へ丸める試作も行ったが、これもGPUと一致しなかった
（`all-rows-long-scalar-ref.log`）。WGSLは積和の融合と丸めの選択を許すため、
このCPUの丸め方だけをGPUの契約とする検証は採らない
（[WGSLの浮動小数点規則](https://www.w3.org/TR/WGSL/#floating-point-evaluation)）。
最初の配列全体の失敗表示が大きくなった実行はSIGINTで止め、要素ごとの表示で切り分けた
（`all-rows-product-focused-v2.log`）。

最終的な追加ケースは、通常GEMMとの全ビット一致を維持した上で、CPUのf64参照からの
前進誤差上界も検査する。乗算・加算・bias・参照の丸め回数から `γ(2K+2) × Σ|項|` を使う。
`γ(n)=nu/(1−nu)`、normalなf32の隣接値への丸めを含む `u=2^-23` とし、
この入力にはoverflow/underflowがない。経験的な閾値の調整ではない。
既存ケースの参照と許容差は変更せず、最適化による退行の検出は厳密なGPU間比較が担う。
隔離の最初の起動はOUTが既定の検出対象外で終了したため、明示したimport mapと`--no-config`で実行した。

重点検証は **92 passed / 0 failed、12秒**（`all-rows-product-focused-v3.log`）。
コミット前の `deno task verify` は **2,886 passed / 760 steps / 0 failed / 5 ignored、24分43秒、終了コード0**
（`all-rows-product-verify.log`）。GPUベンチ・別のGPUテストは並走させていない。
最終の生成器も測定候補との全文一致を確認した（`all-rows-product-source-check.log`）。
追加リンクと見出し、性能台帳の対象外の行が不変であることも検査した（`all-rows-doc-links.log`）。

## topk k=1 の分割と小出力の実験（2026-09-11）

この節はRTX 3080 Ti / Deno 2.9.6 / Chrome 153での時点実測。生データのOUTは
`outputs/bench/karume/2026-09-11_fable-followup/`。M2の実測ではない。

### カーネルの変更

大語彙のtopk(k=1)を、既存argmaxと同じ部分最大→mergeの2 dispatchへ分割する。
最大の値と最小indexを選ぶ規則は変えず、値は元入力のu32をそのまま写す。
mergeで元入力を読むため、部分結果だけでなく元入力の寿命も見積りと実行へ載せる。
新しい公開APIやIRは追加しない。決定は
[ADR 0068 追記9](../decisions/0068-decode-exit-multi-output.md#追記-92026-09-11-topk-k1-の長い行も-2-dispatch-へ分割する)。

`top1-kernel-bench.ts` / `.json` は同じcompute pass内の100反復を、heater後に8試料・順序交替で測った。
基準と候補の値・indexのu32一致を確認してから計測する。中央値は次のとおり。

| 行数 |   行長 | 既存topk k=1 |   分割候補 | 速度比 |
| ---: | -----: | -----------: | ---------: | -----: |
|    1 |  16384 |   0.16208 ms | 0.01004 ms |  16.14 |
|    8 |  16384 |   0.17031 ms | 0.01057 ms |  16.11 |
|   64 |  16384 |   0.17722 ms | 0.01184 ms |  14.97 |
|    1 |  16385 |   0.16867 ms | 0.01029 ms |  16.40 |
|    8 |  16385 |   0.16981 ms | 0.01051 ms |  16.16 |
|   64 |  16385 |   0.16974 ms | 0.01208 ms |  14.06 |
|    1 | 262144 |   2.01720 ms | 0.00922 ms | 218.74 |
|    8 | 262144 |   3.17251 ms | 0.02289 ms | 138.59 |
|   64 | 262144 |   3.21206 ms | 0.09917 ms |  32.39 |

これはカーネル反復の値であり、生成全体の速度比ではない。
既存argmaxのWGSLは全文不変で、k>1と短い行のtopkも維持する。
区間数がWebGPUの既定dispatch上限65,535を超える巨大な行は、既存1 dispatchへ残す。

### 小出力化を含む生成の対照

`small-output.ts` / `small-browser.ts` はQAT E2Bの同じ入力・重みで、実験用グラフの出力だけを
logits+hiddenからtopkの値+indexへ変える。容量128、chunk32、PLE常駐0、温度0、penalty/bias無し、
最大64token、3入力×各3生成。毎回新しいGenerationContextを作るため、実際の多ターン会話ではない。
通常CLIの変更でもない。出力は1tokenあたり **1,054,720 B→8 B**。

既存topkをそのまま使った初期候補はDenoで約5%遅くなった（`small-top1-0-retry.log`、`warm-summary.json`）。
分割カーネルを使った最小差分を、基準→候補→候補→基準の順で比較した。
以下は暖機後6生成の「基準の所要時間÷候補の所要時間」。

| 環境   | 比較組 | 最小 / 中央 / 最大    |
| ------ | ------ | --------------------- |
| Deno   | 1      | 1.007 / 1.012 / 1.016 |
| Deno   | 2      | 1.014 / 1.020 / 1.026 |
| Chrome | 1      | 1.006 / 1.032 / 1.066 |
| Chrome | 2      | 1.007 / 1.063 / 1.090 |

全4組・各9生成のtoken列が一致した。集計は`top1-minimal-model-summary.json`、個票は
`{deno|chrome}-small-minimal-{base|top1}-{0|1}.json`。
出力転送とCPUの語彙走査を減らせば両環境で利益はあるが、Denoの待ち時間の下限は残る。
通常版、E4B、M2での小出力化は未検収。

### 正しさと残る統合

候補のGPU検証は **3 passed / 0 failed、490 ms**（`top1-minimal-gpu.log`）。
行長16,383 / 16,384 / 16,385 / 262,144、8行でNaNの符号とpayload・同点・±0・全−inf・末尾を検査した。
`neg→topk`の中間入力を使うグラフでも、2回の値変更とbacking再利用で値・indexが一致。
workspaceの見積りは実測のピーク以上。部分結果を見積りから外す故障注入は失敗した
（`top1-intermediate-estimate-no-temp.log`）。元入力のmerge読みを省く注入は、現行plannerが
ノードの終わりまで入力を保持するため失敗しなかった。この後者を検出済みとは主張しない。

製品の小出力化には、次の段階が必要。

1. バッチの終端で常駐出力をstagingへコピーし、mapを完了待ちとして使う。
   現在の`BatchScope.finish()`後に`ResidentTensor.read()`を呼ぶだけでは待ちが2回になる。
2. `Session.enqueue`へGenerationContextを渡すなら、発行からバッチの最終成功/失敗まで使用リースを保つ。
   長さのadvance/defer、state書き込み後のpoison、abort・dispose・rewindとの排他を通常runと揃える。
   enqueue本体のin-flightリースと同じ完了待ちにすると自己待ちになるため、寿命を分ける。
3. modelsの内部能力として、温度0・penalty/bias無し・非投機の小出力を明示する。
   最初のNaNと非有限の最大値を現行sampler同様に拒否する。仮の全logitsへ膨らませるadapterは作らない。

公開API・会話状態に触れるため、この段階はカーネル最適化と分けて検収する。
この時点では未実装で、CLIの出力転送を減らしたとは報告しない。

追加でsignaling NaNを正負1行ずつ含む10行へ拡大し、旧topkと候補の両方で値ビットを確認した。
旧側は対象1件成功・2件filter、候補は3件成功（`top1-signaling-baseline.log` /
`top1-signaling-candidate.log`）。製品の重点検証は **149 passed / 5 steps / 0 failed、2秒**
（`top1-product-focused.log`）。既存argmaxのsnapshotは更新していない。

全体検証中の差分レビューで、外側の次元が0の場合に分割用一時が0Bになる退行を見つけた。
初回の全体検証は自分のテストプロセスへSIGINTを送って中断した（`top1-product-verify.log`、
終了コード139。成功した全体検証としては扱わない）。
旧経路は3形状×2実行の空出力を返し、候補は分割境界で0B一時の検査に失敗した
（`top1-empty-baseline.log` / `top1-empty-candidate.log`）。
行数0は既存の1 dispatch経路へ残し、受理範囲を保つGPU回帰テストを追加した。
修正後の重点検証は **150 passed / 5 steps / 0 failed、2秒**（`top1-product-focused-v2.log`）。
全体の `deno task verify` は **2,891 passed / 760 steps / 0 failed / 5 ignored、24分37秒**で成功
（`top1-product-verify-v2.log`）。この全体走行中はGPUベンチ・追加GPUテストを並走させていない。

## 定常速度の追加候補と採否（2026-09-11）

この節はRTX 3080 Tiでの時点実験。OUTは `outputs/bench/karume/2026-09-11_fable-followup/`。

### 重み容量と演算構成

`weight-layout-summary.py` / `.json` で現在のE2B資産を数えた。PLEを除く主グラフのshard合計は
通常版1,579,589,924 B、QAT828,096,932 B。linearは双方277個だが、通常版はINT4が276・INT8が1、
QATはINT2が61・INT4が145・INT8が70・f32が1。QATにはさらにstatic_quantizeが487個ある。
容量は約半分でも演算と同期の費用は残るため、容量比からtok/sの倍率は予測できない。
ノード数は時間の帰属ではない。利用者のM2ではDenoの通常版が初回6.8、2回目16.4 tok/s、
QATが初回4〜5、2回目16〜17 tok/sとの観測で、これをRTXの実測と混同しない。

### 数値を変えないINT2候補

`shared-x-i2.ts` / `.json` は入力をworkgroupメモリへ共同で読み、Kタイル256/512/1024/4096を比較した。
実重みdown/up/head×3入力で、rawと量子化後のu32は基準と一致した。
速度比の最良はdown0.936、up0.986、head1.034。大きなタイルではheadも0.788へ低下した。
共通適用には利益がなく採らない。集計は`shared-x-summary.json`。

4値の逆量子化を先に計算して整数コードで選ぶ候補は、単体downで1.086倍だった。
`select-import-map.json`はM=1・K>=4096・N<=4096だけに限定してモデル全体を比較したが、
暖機後の中央値はDeno1.001 / 1.009、Chrome0.992 / 1.005で、安定した利益がなかった。
全4組・各9生成のtoken列は一致した。採用せず、`select-model-summary.json`へ記録した。

列ごとに連続する重みを並べ替え、隣接スレッドの読みをまとめる候補は、raw値が一致し、
down1.014・up1.006・head1.047倍。headの並べ替えだけでも約96 msを要し、
保存形式やアップロードの複雑さに見合う全体利益を確認できていないため採らない。
集計は`exact-micro-summary.json`。小workgroupとM=1関数化の不採用理由はINT2行ブロック節を参照。

### K方向の並列縮約

`parallel-i2.ts` / `.json` は語の加算順を変える別候補。8レーンのf32部分和はdownで2.777倍、
up0.916・head0.599倍だった。down相当の形だけに限定したQAT E2Bは
暖機後1.035〜1.046倍で、9生成のtoken列は基準と一致した（`qat-parallel-0.json`）。
ただしraw u32と一部の量子化後の値は変わる。

`parallel-oracle.ts` / `.json`で独立CPUの投影結果との距離も比較した。
downのsin / wide / near-boundary入力で、量子化前のmaxAbsは順に
5.96e-6→2.86e-6、5.064e-4→1.106e-4、6.866e-5→3.338e-5へ改善した。
一方、sin入力の量子化後にCPUと異なる要素は1/1536→2/1536へ増えた。
丸め境界を跨ぐため、量子化前の誤差改善が量子化後の一致を保証しない。
整数16レーンではこの3入力の量子化後差が0だったが、入力scaleを知る新しい実行契約が必要。
単純な浮動小数点誤差上界で境界を避ける案は93〜100%の出力が判定不能となり、不採用。

既定経路には入れない。別の明示的な数値モードとして扱う場合も、E4B・M2・複数入力と長い生成の
品質評価が必要。M=1だけの変更は、投機verify行0とdecodeのu32一致を保証しない。
QATのMTPは未対応であり、将来の統合へそのまま適用できるとはしない。

### 公式CPUの長い生成との比較

`e2b-cpu-long-reference.json` / `long-reference-summary.json` は同じ3入力を最大64tokenまで比較した。
GPUの変更前・非同期準備後・並列縮約候補は同じprefix一致長で、code46/64、物語13/64、日本語0/18だった。
日本語のGPU生成長は20token。これらは今回の最適化より前からある差で、
GPUの基準とのtoken一致を「公式CPUと64token完全一致」とは報告しない。

### 量子化の固定回数探索とlinearとの融合

`srq-search-bench.ts`はstatic_quantizeの値・境界表を変えず、while探索を固定8比較へ展開した候補。
CPUの802,723点で従来探索と一致し、公式30scale・24,416入力を含む既存GPU検証4件も成功した
（`srq-search-cpu-check.json`、`srq-branchless-oracle.log`）。
6サイズ×3分布、100反復×8試料・順序交替の単体測定では、全18条件で1.018〜1.591倍だった
（`srq-search-bench-summary.json`）。
しかしQAT全体のABBAでは、暖機後中央値がDeno0.990 / 1.006、Chrome1.001 / 1.009で利益が安定しなかった。
全4組・各9生成のtoken列は一致。採用しない（`srq-model-summary.json`）。

`linear-srq-census.json`では、277個のlinearのうち276個が直後のstatic_quantizeだけに消費される
（INT4 145・INT8 70・INT2 61）。量子化scaleが0で恒等になるopは487個中2個に留まる。
`linear-srq-fusion-bench.ts`はINT2のM=1の末尾へ同じ整数表の量子化を付け、2 dispatchを1つにした。
down/up/head×3入力×bias有無の18条件で量子化後u32が基準と一致した。
ただし単体の速度比中央値はdown0.971・up1.050・head1.019で、downは遅くなった。
サンプル間の揺れもあり、この形での共通適用は見送る。集計は`linear-srq-fusion-bench-summary.json`。
INT4・INT8への融合やM2の速度は、この実験では評価していない。

### 推定位置を固定回数で補正するSRQ候補

`srq-hint-candidate`は、前回のQAT統合後に試した商からの境界探索と同系列で、新しい原理とは扱わない。
今回は上下1段の補正後に整数表の区間を検査し、未確定なら従来二分探索で確定する。
35scaleのうち恒等の0を除く34scale・境界前後・ランダムビットと、推定値の±2段故障注入を含む
**10,256,541比較**が一致した（`srq-hint-cpu-check.json`）。
公式30scale・24,416入力を含む既存GPU検証も **4 passed / 0 failed、2秒**（`srq-hint-oracle.log`）。

同じ6サイズ×3分布の単体速度比は0.744〜1.412倍で、混合分布では全6サイズが遅く、
262,144要素と1,048,576要素では約0.744 / 0.748倍だった。
`srq-hint-bench.json` / `srq-hint-bench-summary.json`へ保存した。
範囲による損が大きく、過去の同系列候補でも全体利益が安定していなかったため、
今回の候補は全モデル比較へ進めず不採用とした。

## バッチ終端の一括読み戻し（2026-09-12）

この節はRTX 3080 Ti / Deno 2.9.6 / Chrome 153での時点実測。OUTは
`outputs/bench/karume/2026-09-11_fable-followup/`。M2の実測ではない。

### 読み戻しの順序と公開面

`BatchScope.finishAndRead(outputs)`を追加する。全enqueueのエンコード後、指定した常駐出力を
1個のstagingへコピーし、mapを唯一の完了フェンスにする。既存の`finish(): Promise<void>`と
`ResidentTensor.read()`は維持する。指定は発行時に固定し、使用予約はバッチの最終決着まで保つ。
空の集合は既存のqueueフェンス、別device・破棄済み・合計上限超過・重複指定は明示拒否する。
決定は[ADR 0054](../decisions/0054-resident-loop-and-fence.md#バッチ終端の一括読み戻し2026-09-12)。

`batch-read-bench.ts`は、2個の既存residentを`finish→並行read`で読む経路と、一括読み戻しを比較する。
4回の準備後、8試料×各10回を順序交替で測定した。計測器で全試料の
queueフェンス1回/map2回→queueフェンス0回/map1回を確認した。
これは転送プロトコルの対照であり、既に単一フェンスの`Session.run`に対する倍率ではない。

| 環境   |    出力合計 | finish後にread | 一括読み戻し |
| ------ | ----------: | -------------: | -----------: |
| Deno   |         8 B |      22.437 ms |    11.197 ms |
| Deno   | 1,048,580 B |      23.253 ms |    12.076 ms |
| Chrome |         8 B |       1.081 ms |     0.527 ms |
| Chrome | 1,048,580 B |       1.577 ms |     1.383 ms |

数値の正本は`batch-read-bench-early-summary.json`、個票は`batch-read-bench-{deno|chrome}-early.json`。
初期案はmap後にerrorScopeを検査し、Chromeの大出力が1.587→3.384 msへ悪化した
（`batch-read-bench-summary.json`）。copyのsubmit直後にerrorScopeを検査し、その後mapする
既存runと同じ順序へ直すと、上表では退行が解消した。Chrome内部の遅延理由まで帰属したとはしない。

追加した13条件は、非await enqueue、memberの写し、12B境界・重複名・NaNの値ビット、
破棄/別device/不正型、空集合、finish/settle競合、確保/copy/map/読み出し失敗、device loss、
GPU失敗とホスト失敗の原因保持を検証する。
製品版の重点検証は **45 passed / 0 failed、7秒**（`batch-read-product-focused.log`）。
全体検証は **2,904 passed（760 steps）/ 0 failed / 5 ignored、24分58秒**
（`batch-read-product-verify.log`）。

### 会話状態を使う次段階の試作

`batch-generation-candidate`は`enqueue.generation`と終端時の長さ確定を加えた隔離コピー。
使用予約を終端まで保ち、成功時だけadvance/defer、state書き込み後の失敗ではpoisonする。
バッチ中のdisposeを明示拒否する案で、通常runの破棄待ち契約は維持する。
既存98件と追加11件を実行し、追加1件の「空のsliding履歴でrewindできる」という誤った前提を訂正した。
製品の既存テストは変更していない。既存98件・追加10件成功/1件失敗のログは
`batch-generation-focused.log`、訂正した追加全11件の成功は`batch-generation-focused-v2.log`。
借り手の予約伝播、full/slidingのprefillとdecodeのu32一致、後続失敗と複数contextの確定失敗も含む。
この段階は製品未統合。

`batch-model-deno.ts` / `batch-model-browser.ts`は保存済みQAT E2B資産を変更せず、
targetのlogitsを常駐出力へコピー→別topk Session→8B読み戻しで実行する。
容量128、chunk32、PLE常駐0、温度0、penalty/bias無し、最大64token、3入力×各3生成。
毎回新しいcontextを作るため実際の多ターン会話ではなく、CLIの計測でもない。
基準→候補→候補→基準の比較で、全4組・各9生成のtoken列が一致した。
1tokenあたりのCPU出力は1,054,720 B→8 B。暖機後6生成の速度比中央値は次のとおり。

| 環境   | 1組目 | 2組目 |
| ------ | ----: | ----: |
| Deno   | 1.015 | 1.032 |
| Chrome | 1.052 | 1.027 |

正本は`batch-model-summary.json`、個票は`{deno|chrome}-batch-model-{base|batch}-{0|1}.json`。
実グラフを書き換えない経路にも利益はあるが、一般samplingのCLI既定設定の高速化ではない。
次はgenerationの寿命を製品へ統合し、modelsのgreedy能力と多ターン・中断・resetを検収する。
通常版/E4B/M2、温度あり・penalty/bias・投機の小出力化は別途必要。

## 会話状態のバッチ統合（2026-09-12）

この節はRTX 3080 Tiでの時点検証。OUTは前節と同じ。M2の検収ではない。

`Session.enqueue`へ`generation`を追加し、既存のcontext・queryLength・commitを受ける。
入力の写しと使用予約は発行時、論理長のadvance/deferと予約返却はバッチ最終決着時とする。
後続Sessionや読み戻しの失敗も書き込み済みcontextをpoisonする。IRと保存資産は変えない。
[ADR 0066](../decisions/0066-generation-context-state-slots.md#バッチ実行の-generationcontext2026-09-12)
に、借り手への予約伝播・通常runとバッチ中の破棄の違いを記録した。

バッチの使用予約中はcontext/所有Sessionのdisposeを受付終了前に拒否する。
単にdisposeをSessionの操作鎖へ追加すると、前のenqueue→dispose→後のenqueueの順で、
disposeがバッチ完了を待ち、バッチが後のenqueueを待つ循環になる。
従来runの二段破棄は維持し、バッチ完了後の破棄を受け付ける。

11件の追加検証はfull/slidingのu32一致、途中settleと最終確定、発行指定の写し、
並行run/rewind/dispose拒否、deferred確定、書き込み前後の失敗、複数contextの確定失敗、
不正指定の部分予約解放、借り手と貸し手の寿命を対象にする。
製品版は既存の状態/借用/バッチ/フェンス検証と合わせて **96 passed / 0 failed、17秒**
（`batch-generation-product-focused.log`）。lint・型検査も成功した。

modelsの隔離試作では、既存CPU検証118件（54 steps）と追加7件、GPU追加3件が成功した。
記録は`model-greedy-existing-cpu.log` / `model-greedy-new-cpu-v3.log` /
`model-greedy-gpu-test.log`。実pipeline比較のDeno QAT E2B・通常E2B・QAT E4Bでも
生成token・多ターン・温度切替・並行会話・中断再開・reset後の結果が一致した。
通常E4Bの初回は比較器のmanifest参照先がE2B専用だったためロード時に失敗した
（`pipeline-greedy-deno-normal-e4b-base.log`）。既存E4B実験資産へ参照先だけを直し、
`-v2`の別出力で再試験し、Deno/Chrome・通常/QAT・E2B/E4Bの全8条件で結果が一致した
（`pipeline-greedy-smoke.log` / `pipeline-greedy-rest.log` / `pipeline-greedy-remaining.log`）。
初回比較の基準側は経路固定に診断callbackを使ったので、速度の採否には使わない。
診断なしの旧modelsコピーと候補を改めて正順/逆順で比較する。
これらはmodelsの製品統合前の検証で、CLI速度の最終判断ではない。

主担当の再レビューで、不正contextの内部参照が常駐出力の使用予約後に例外を出す経路を発見した。
追加テストは修正前に使用予約1本の残留で失敗した（`batch-generation-invalid-context-before.log`）。
内部参照を予約前の検査区間へ移し、追加1件を含む **97 passed / 0 failed、17秒**を確認した
（`batch-generation-product-focused-v2.log`）。初期の11件にこの回帰1件を加え、製品の追加検証は12件。

modelsの診断なし正順/逆順比較も全16組で出力が一致した（`pipeline-greedy-clean-summary.json`）。
ただし、初回prefillから小出力を使う案ではChrome通常E4Bの最初のtoken到着が
535→639 ms / 517→663 msと遅くなった。driver cacheを管理していないため原因の帰属は未確定だが、
初回runのアリーナ経路を初回enqueueのbacking構築へ置き換える差もある。
この初回経路は採用せず、prefillを既存runで維持し、decodeだけを小出力にする別候補へ進む。
`model-greedy-decode-candidate`のCPU検証は **125 passed（54 steps）/ 0 failed、552 ms**
（`model-greedy-decode-cpu.log`）。実モデルと製品統合の検収は後続段階。

会話状態のバッチ拡張の全体検証は **2,916 passed（760 steps）/ 0 failed / 5 ignored、24分44秒**
（`batch-generation-product-verify.log`）。models側の小出力経路はこの検証・コミットには含めない。

## 最大値選択の非正規数対応（2026-09-12）

この節はRTX 3080 Ti / Deno 2.9.6 / Chrome 153での時点検証。M2の実測ではない。
出力ディレクトリは全て `outputs/bench/karume/` の下で、今回から実験ごとに新規作成する。

### 再現と比較方式

小出力生成を統合する前の境界プローブで、現行topkが非正規数をゼロと同値に比較し、
10条件中8条件でCPUと違う添字を選ぶことを確認した
（`2026-09-11_fable-followup/top1-extreme-probe-deno.json`）。
`2026-09-12_rank-bits-9y95mp76/baseline-regression.log`でも、
新設したargmax/topkの境界テストと一般kテストが両方失敗した。旧版の参照先も固定して再現している。

[WGSL §15.7.2](https://www.w3.org/TR/WGSL/#floating-point-evaluation)は比較演算などで非正規数のゼロ化を許す。
比較直前のbitcastだけでは途中値のゼロ化も残るため、入力・scratch・部分結果までu32で保持する。
公開API・IR・保存資産は変えず、既存の順位と値ビットを守る修正とする。

最初の案は比較のたびに整数順へ変換する。次の案は入力時に一度だけ順位キーへ変換する。
正値の符号ビットを反転し、負値は全ビットを反転する。±0は同点、NaNは全て最優先の同一キーとし、
同点なら最小indexを採る。値を返すtopkは元入力から選択値のビットを再読する。
採用案は後者（[ADR 0068追記10](../decisions/0068-decode-exit-multi-output.md#追記-102026-09-12-非正規数を失わない最大値上位候補の比較)）。

### 数値検証

- `2026-09-12_rank-key-pgcyshci/cpu-proof.json`: 通常のCPU数値比較に対して3,000,768組が一致。
- 両案とも既存演算・分割・新設2件のGPU検証が **45 passed / 0 failed、16秒**。
  `2026-09-12_rank-{bits-9y95mp76|key-pgcyshci}/gpu-focused.log`。
- 両案とも生成物・分割条件は **79 passed（5 steps）/ 0 failed**（同ディレクトリの`cpu-tests.log`）。
- Chromeでも両案の境界・一般kの2検証が成功（同ディレクトリの`chrome-probe.json`）。
  実GPUのvendor=nvidia・architecture=ampere・fallback=falseも記録した。

新設検証は通常のCPU比較を基準にし、GPUの整数変換を複製しない。
短い行、分割境界16,384の前後、語彙262,144、k=4/16/63、
正負の非正規数、NaN payload、符号付きゼロ、Infinity、同点を扱う。
既存の許容誤差・goldenは変更しない。同点方向・有限sentinel・NaN優先順位の故障注入を維持する。
生成物の変更に合わせてキーをv2、WGSLスナップショット6件を更新する。

### 単体性能と計測の対照

最初は各候補の前に既存heaterを実行し、ホストへ戻ってから100反復を計測した。
27条件×三者×9試料・順序循環の値と添字は一致したが、大きい行数などで時間が二分した。
argmaxの行長262,144・64行では順位キーが0.666倍、対応するtopkでは1.615倍となった。
個票は約0.10/0.16 msの二群で、中央値だけの採否は不適切だった
（`2026-09-12_rank-bits-9y95mp76/kernel-threeway.json`）。

`2026-09-12_rank-contiguous-tge0tn56/bench.ts`では、同じコマンド投入内で約40 ms相当以上の暖機を行い、
三者の計測passをホスト待ち無しで続ける。100反復×12試料、三者の全6順序を2巡した。
全27条件で値・添字が一致し、順位キーの速度比は **1.037〜1.299倍、中央値1.088倍**だった。
比較のたびに変換する案は0.847〜0.967倍で、こちらは採らない。
先の二群の原因をGPUクロックと断定はしないが、投入方法を揃えた対照では退行は再現しなかった。

| 演算   |    行長 | 行数 |  k |        旧版 |    順位キー |
| ------ | ------: | ---: | -: | ----------: | ----------: |
| argmax | 262,144 |    1 |  1 | 0.008796 ms | 0.008083 ms |
| argmax | 262,144 |   64 |  1 | 0.102898 ms | 0.099098 ms |
| topk   | 262,144 |    1 |  1 | 0.009000 ms | 0.008270 ms |
| topk   |  32,768 |    1 | 16 | 1.368187 ms | 1.053175 ms |

正本は同ディレクトリの`kernel-threeway.json`と`kernel-threeway-summary.json`。
基準コードは`2026-09-12_rank-bits-9y95mp76/baseline/runtime`へ凍結し、
`baseline-code.json`と`*-frozen.json`のimport mapで製品修正後も追試できるようにした。
これはカーネル単体の倍率であり、CLIの生成速度の倍率ではない。

実験準備で誤上書きした旧`prepare-top1-bits.py`は、利用者の復元指示を受け、作成時の実行履歴から復元した。
後続のcat履歴ともバイト一致を確認し、復元元とハッシュを
`2026-09-11_fable-followup/incident-2026-09-12T0135-script-overwrite/recovery.json`へ保存した。
旧スクリプト自体は再実行していない。既存モデル・以前の測定結果・製品コードへの影響は無い。
専用ディレクトリと排他的作成、準備失敗時の後段中止はAGENTS.mdへ永続化した（`053c87e`）。

製品版の重点検証は **124 passed（5 steps）/ 0 failed、17秒**
（`2026-09-12_rank-key-pgcyshci/product-focused.log`）。
全体検証は **2,918 passed（760 steps）/ 0 failed / 5 ignored、24分45秒**
（同ディレクトリの`product-verify.log`）。この検証・コミットにはmodelsの小出力候補を含めない。

## Gemmaの温度0decodeの小出力化（2026-09-12）

この節はRTX 3080 Ti / Deno 2.9.6 / Chrome 153での時点検証。M2の実測ではない。
主実験は `outputs/bench/karume/2026-09-12_greedy-after-decode-oj136f8y/`（以下OUT）。
追試は同じ親の `2026-09-12_greedy-repeat-qky5qjyx/`。

通常Gemma4とGemma4-QATの共通pipelineに、温度0・非投機・penalty/bias無し・診断無しの
**decodeだけ**、target enqueue→常駐logits→小さなtopk Session→8B読戻しを接続した。
prefillは従来runを維持する。保存済みlogits+hiddenのグラフ・資産・CLI引数は変更しない。
最大値・NaNの最小添字・非有限値の拒否は従来CPU選択と同じ。一般samplingと投機は従来経路。
[ADR 0083](../decisions/0083-generation-api-surface.md#gemmaの温度0生成の小出力2026-09-12)に
適用条件、通常runとbatchの直列化、準備失敗と資源解放を記録した。

### 比較条件と一致

基準は小出力統合前のmodelsコピー、runtimeは両者とも`50675cc`の整数順位キー修正後。
診断callbackは両方とも無し。Denoはローカル位置読み、ChromeはlocalhostのHTTP Range。
容量128・chunk32・PLE常駐0・温度0・最大64token、3入力を各3回生成する。
別途、同じsequenceの温度0→1→0、温度の異なる並行会話、decode後のabort/breakと再開、
Gemma4ChatSessionの2ターンと作り直し後の再現を比較した。
旧検証器は最初のtoken直後の中断でprefillしか通らなかったため、2token受信後へ変更した。
CPU検証では小出力decodeを実際に1回通ったことも検査する。旧準備フォルダは残し、上書きしていない。

Deno/Chrome × 通常/QAT × E2B/E4Bを正順・逆順で比較し、**全16組が生成列・停止結果・会話状態で一致**。
`pipeline-greedy-decode-summary.json`と`pipeline-greedy-*-decode-{forward|reverse}.json`が正本。
各候補はdecodeごと8Bのmapを確認した。これは温度ありのCLI既定設定の速度比較ではない。
Apple GPUや公式CPUとの全生成一致を、この比較から主張しない。

隔離候補のCPU検証は **125 passed（54 steps）/ 0 failed、556ms**（`cpu-tests.log`）。
GPUの通常runとの混在・部分確保失敗後の再試行・未await生成を待つdisposeは
**3 passed / 0 failed、682ms**（`gpu-tests.log`）。初回統合の6ファイルは検証候補と同じで、
`source-manifest.json`に測定対象のハッシュを保存した。後述の見積り内訳のみ、全体検証後に修正した。

### 暖機後の速度とばらつき

各入力の2・3回目、計6生成の壁時間比の中央値。1より大きいと候補が速い。
モデル読込を除き、prefillから生成完了までを含む。最初のtokenを除いたdecode tok/sとは区別する。

| 環境   | モデル | サイズ |   正順 |   逆順 |
| ------ | ------ | ------ | -----: | -----: |
| deno   | 通常   | E2B    | 1.0211 | 1.0174 |
| deno   | 通常   | E4B    | 0.9928 | 1.0005 |
| deno   | QAT    | E2B    | 1.0178 | 1.0111 |
| deno   | QAT    | E4B    | 0.9900 | 1.0221 |
| chrome | 通常   | E2B    | 1.0706 | 1.0532 |
| chrome | 通常   | E4B    | 1.0334 | 1.0412 |
| chrome | QAT    | E2B    | 1.0692 | 0.9976 |
| chrome | QAT    | E4B    | 1.0156 | 1.0241 |

Chrome QAT E2Bは正順約7%改善に対して逆順が横ばいだったため、同条件で4組を追加した。
全て一致し、壁時間比は **1.0698 / 1.0876 / 1.0399 / 1.0436**。
主実験の横ばい結果を除外せず残す。ばらつきの原因は特定していないが、追加4組で利益を再確認した。
Deno通常E4Bの初回token到着は主実験正順254→503ms、逆順253→248msだった。
追加2組は250→246ms / 247→245msで、最初の外れ値は再現しなかった。
追加2組の暖機後比は **0.9966 / 1.0249**。Deno E4Bの安定した利益は主張せず、ほぼ中立と判断する。
追試の正本は`summary.json`と`{chrome-qat-e2b|deno-normal-e4b}-*-{base|greedy}.json`。
主実験と追試の計**22組・44実行**で出力が一致した。

初回時間はdriver cacheを管理したcold測定ではない。prefillの実行経路を維持しても揺れがあり、
初回が必ず速くなるとは扱わない。追加資源の見積りは全4モデルで **2,097,940 B**。
GPU資源は最初のgreedy decodeまで遅延確保し、見積りは借用入力のぶんを含む保守的な上界。

Chromeを中心とする利益と結果一致に基づき、温度0decodeの候補を採用する。
M2での初回/暖機後の追試、温度あり・penalty/bias・投機の転送削減は残る。
`deno task demo:gemma4-qat --model e2b --temperature 0`で既存ローカル資産を使って試せる。

製品版の重点検証は **128 passed（54 steps）/ 0 failed、1秒**
（OUTの`product-focused.log`）。許容誤差・golden・既存の期待列は変更していない。

### 全体検証で見つかった見積り内訳の修正

初回の全体検証は **2,927 passed（760 steps）/ 1 failed / 5 ignored、25分19秒**
（OUTの`product-verify.log`）。失敗は`e2e_gemma4_chat_test.ts`の予算透過検証で、
単独でも **2 passed（12 steps）/ 1 failed、17秒**（`product-chat-isolated.log`）と再現した。
差額2,097,940Bは追加した小出力資源の量に一致する。これはVRAM圧の失敗ではなく、
ピークだけに足して内訳へ載せなかった実装が`AdmissionReport`の合計式を崩していた。

修正の作業先は `outputs/bench/karume/2026-09-12_greedy-memory-k95clxph/`。
公開の省略可能欄`auxiliaryBytes`へ補助資源を別計上し、合計へ加える。runtime単体の既存レポートは欄を省略し、値も変えない。
通常/診断の両経路で予算・シナリオ・重み・stateを照合し、追加勘定とピーク差を厳密に検査する。
投機pipelineの合成後にも内訳が残ることを検証する。既存の等値検査を許容誤差へ緩めていない。
推論・GPU資源の確保量・選択経路は変更せず、実測候補からの差は報告の内訳とその検査だけである。

修正後の会話ファイル全体とruntime見積りは **70 passed（12 steps）/ 0 failed、22秒**
（修正用ディレクトリの`focused.log`）。
投機pipelineの見積り・生成は **1 passed（2 steps）/ 0 failed / 8 filtered out、2秒**
（同ディレクトリの`speculative-report.log`）。これは重点確認で、続く全体検証ではフィルタを使わない。

修正後の全体検証は **2,928 passed（760 steps）/ 0 failed / 5 ignored、24分48秒**
（修正用ディレクトリの`product-verify.log`）。TypeScriptの対象10ファイルは再検証中に変更していない
（`final-source-hashes.json`）。既存のスキップ5件も、数値修正前後と同じ母音検出の資産依存ケースである。

## デモのTTFT分離と起動時ウォームアップ（2026-09-12）

この節はRTX 3080 Ti / Denoでの時点検証。利用者のM2の2ターン目16.9 tok/sという観測を受け、
公開デモと比較するため、まず計測条件を揃えた。ここで測るのはデモの振る舞いであり、新しいカーネルの速度ではない。

Gemma通常/QATの会話オプションに復号前の`onToken(id)`を追加した。停止tokenは通知しない。
TTFTは開始から最初の非停止tokenまで、decode速度は`(配送数 - 1) / (完了時刻 - 最初の配送時刻)`。
0 tokenのTTFT、0〜1 tokenまたは時間0の速度は未測定と表示する。ターン全体時間と停止込みtoken数も残す。
MiniCPM5/Qwen3も同じ算出式へ揃え、JSONには既存のelapsedMsに加えてgeneratedTokens・ttftMs・
decodeTokensPerSecondを出す。未測定はnull。文字列の再符号化や最初の非空文字列では計測しない。

起動時の暖機は既定有効、`--no-warmup`で無効にできる。Gemmaは全chunk長のprefillと最大4tokenの
短い生成を独立sequenceで行い、解放してから会話を開始する。MiniCPM5/Qwen3も独立した短い生成で暖機する。
本番のsampler設定を用いるが、履歴・KV・乱数状態・投機ゲートは本番と共有しない。
全ての入力長・投機経路のコンパイルやdriver cacheの管理を保証するものではない。
契約変更は[ADR 0084](../decisions/0084-gemma-tokenizer-chat.md#トークン通知とデモ計測2026-09-12)。

重点CPU検証は**78 passed（84 steps）/ 0 failed、1秒**。
`outputs/bench/karume/2026-09-12_demo-timing-qpj_8tgs/focused.log`に保存した。
続く型検査はGenerationSequence型にasyncDisposeが無いため失敗し、公開disposeをawaitする
所有ラッパーへ直した。4デモの型検査で修正を確認した。

実GPU比較の初回は読込時間の表示を比較器が除去しておらず失敗した。本文・停止・会話には差がなく、
時刻除去を直して新規ディレクトリで再走した。初回ログは`2026-09-12_demo-warmup-check-eedb5xj3/`に残す。
最終結果は`outputs/bench/karume/2026-09-12_demo-warmup-recheck-n45nwi85/summary.json`。
Gemma通常、同投機、QAT E2B/E4B、MiniCPM5、Qwen3の**6構成・各3ターン・暖機あり/なし**で、
回答・停止結果・reset後の再現が一致した。MiniCPM5/Qwen3はJSONのtoken列・prompt・KV再利用数も一致。
Gemmaの比較は本文と時刻以外の表示を対象にし、token列の完全一致とは主張しない。

温度1・seed42のGemma通常/QAT E2Bでも各3ターンの暖機あり/なしが一致した。
`outputs/bench/karume/2026-09-12_demo-sampling-check-b93o_pel/summary.json`が正本。
同じディレクトリにQwen3単発completionのJSONとMiniCPM5単発chatの表示確認も保存した。
GPUの仕事は全て逐次実行した。Apple GPUでの新表示と暖機の確認は利用者の実機で行う。

全体検証は**2,934 passed（760 steps）/ 0 failed / 5 ignored、24分51秒**。
ログは`outputs/bench/karume/2026-09-12_demo-timing-verify-cjo4gmo1/verify.log`。
その間にGPUの別ジョブは実行していない。終了後、通常Gemmaの既定capacity4096 / chunk768でも
暖機と4tokenの生成を確認した（`2026-09-12_demo-default-warmup-yiy7_l_e/`）。
暖機0.87秒・TTFT142msはこの単発の動作確認値で、安定した性能改善の根拠には使わない。

## LLM の品質参考値と速度比較の優先（2026-09-12）

デモの暖機・TTFT分離に続き、[PyTorchの品質参考値](2026-09-12-llm-quality-baseline.md)を作成した。
Qwen3 / MiniCPM5は元重みと保存済みGPTQの4構成を固定64問と短い文章で計測済み。
Gemma通常/QATのE2B/E4Bは読み込みと少数採点まで成功した。量子化の値と公式QATの丸め条件も照合した。
利用者の主目的はtok/s比較であり、品質評価だけに作業の軸を移したのは解釈違いだった。
利用者は品質計測にも意味があるとして保持を希望したため、完了分を記録し、速度比較を優先する。
CPU品質評価の所要時間をGPUのtok/s比較に流用しない。

## LLM の速度ベースライン（2026-09-12）

[速度比較の正本](2026-09-12-llm-speed-baseline.md)に6モデルのTTFT・暖機後tok/sを記録した。
入力token・停止・量子化重みを固定し、Deno/WebGPUとPyTorch/TransformersをRTX 3080 Tiで逐次実行した。
通常モデルのPyTorchは保存整数をdense f32/BF16へ展開するため、格納方式と追加の丸め条件を明記する。
QATは公式multimodal wrapperの全埋め込み展開を切り分け、同一量子化層を公式CausalLMへ接続する
text条件を追加した。CPUのprefill/cached decode logits一致と、E2BのCUDA生成列一致を確認した。

暖機後はQwen/MiniCPMにPyTorchとの差があり、通常/QAT Gemmaではこの条件のDenoが速かった。
とくにMiniCPM/QwenのTTFT差を次の調査候補とする。全行logitsの投影・転送が候補だが、時間の帰属は未完。
QATと一部BF16の生成列はDenoと分岐したため、速度の同条件比較と出力の一致を区別して記録した。
M2の同条件測定、長文脈、Gemmaの広い品質評価は残る。デモ・品質・速度を作業単位で分けてコミットする。

速度基準値の統合後の全体検証は**2,934 passed（760 steps）/ 0 failed / 5 ignored、25分20秒**。
ログは`outputs/bench/karume/2026-09-12_llm-speed-verify-fNWKU5/verify.log`。
Pythonの時計・品質復元/採点は22件成功。反復生成160件、資産93ファイルのSHAと記録の参照先も照合した。

## Gemma E2B / QAT E2BのChrome比較（2026-09-12）

[ブラウザ比較の正本](2026-09-12-browser-llm-speed.md)と[集計JSON](2026-09-12-browser-llm-speed-results.json)を追加した。
`deno task bench:llm-browser`でローカルサーバーを起動し、M2のChromeから初回・暖機後のTTFT / tok/sを確認できる。
通常 / QAT E2B、karume / Transformers.jsを独立iframeで逐次実行し、モデル読込時間と生成時間を分離する。
既存karume配布物を読み取り専用で使い、公開ONNXは固定revisionから取得する。ローカルONNXの指定も可能。

RTX 3080 Ti / Chrome153で、ローカルONNX・HF直接取得の計80生成が各5反復でtoken一致した。
HF直接取得時の英語中央値は通常karume **51.07 tok/s** / Transformers.js **11.71 tok/s**、
QATは **52.78 / 12.65 tok/s**。karumeの通常/QATの生成列は前回のDenoとも一致した。
ONNXの量子化・演算精度・KV構成はkarumeと異なり、エンジン間の出力は分岐する。ORTの速度ばらつきも残る。
WebML Communityの専用カーネルを動かした結果とは扱わない。

headlessでadapterが得られないため、Xvfb上のheaded Chromeを自動操作した。
Linux / NVIDIA向けf16実験toggleを明示し、M2の通常起動へは持ち込まない。
M2での新ページの速度と、広い品質評価は利用者の実機追試・後続作業として残る。

比較ページ統合後の全体検証は**2,935 passed（762 steps）/ 0 failed / 5 ignored、25分13秒**。
ログは`outputs/bench/karume/2026-09-12_browser-verify-9UMwHy/verify.log`。
JSON保存の画面操作と、最終bundleが計測時と同一であることも確認した。

## M2 のブラウザ実測と PLE 行キャッシュ（2026-09-12）

利用者の通常/QAT E2BのM2実測を確認し、[調査の正本](2026-09-12-ple-row-cache.md)と
[M2の保存結果](2026-09-12-m2-browser-speed-results.json)、[RTXでの改善前後](2026-09-12-ple-row-cache-results.json)を追加した。
HTTP RangeのPLE入力準備に固定費が残ることを帰属し、通常/QAT共通loaderで量子化行を最大256行だけ再利用する。
既存ホスト予算の空きのみを使い、全量shard優先・予算0で無効。値とscaleをそのまま保持し、既存の逆量子化を使う。
公開引数・保存資産・GPUカーネル・精度は変更しない。内部診断の保持byte数は行の保持も数える（ADR 0085追記）。

未使用の英語/日本語6入力でChrome E2B/E4BのTTFTとdecodeが改善し、Deno E2Bはほぼ中立。
全20実行・240生成が120組の基準/候補でtoken列と停止結果が一致した。CPU重点検証は34 passed（38 steps）。
細かい入力バケットも独立実験で効果があったが、既定は変更していない（H-23）。
M2改善後、CacheStorageのBlob経路、長文、MTPの速度は残件。QATのM2/RTX生成列の差は導入前のbundle同士にあり、
同じ機器内の反復一致と区別して記録した。今回の変更がその差を解消したとは扱わない。

全体検証はベンチ・サーバー停止後に実施し、**2,936 passed（771 steps）/ 0 failed / 5 ignored、24分53秒**で成功。
ログは `outputs/bench/karume/2026-09-12_ple-row-cache-verify-b49feca6/verify.log`。
最終browser bundleと計測対象のbyte一致、実測と生成列のhash、文書リンクも確認した。
