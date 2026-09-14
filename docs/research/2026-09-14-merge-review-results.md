> 2026-09-14時点の独立レビューと利用者提供M2計測のスナップショット。未計測条件の品質・性能や、未発見の不具合が無いことを保証するものではない。

# マージ前レビューの結果とM2の再計測

## 判断と範囲

`4dca96af3bee228065c9ece1b0826f23f7822f7f..315732a667b00fbdc747f560a69ed11476420aa2`をレビューし、
今回の範囲では修正が必要な、再現可能な新規不具合を見つけなかった。
[レビュー資料](2026-09-14-merge-review.md)の5分野をSol 3担当と主担当へ分け、差分・関連ADR・呼出元・テストを確認した。
エージェントは実装を変更せず、主担当が返却内容と実コードを照合した。
今回の差分は結果と現況索引の更新のみ。実装・数値契約・モデル既定・資産・main・他ブランチは変更していない。

| 担当                     | 確認範囲                                                                                                                                                             | 結果・限界                                                                             |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Sol: 生成・所有権・batch | 未開始generatorの終了、chatのコピーとcleanup、PLEの破棄競合、Irodoriの多重解放、hub fetch、非同期compile、batch読戻し、GenerationContextの進行・汚染・使用予約       | 新規指摘なし。実device消失と非同期compile失敗の同時発生は今回再現していない            |
| Sol: GPU・数値           | I2/I4/I8 GEMV、端列load、barrier、scale添字、M1/4/8の選択、RMS融合、subgroup、SRQ、topk/argmax、fusion/estimate/weight residency                                     | 新規指摘なし。Chrome/Metalの全形状・実モデルの新たなu32比較はしていない                |
| Sol: QAT・exporter       | 記号view/safe_softmax、固定writer、INT2形式と語彙、SRQ、PLE schema 2、QAT構造門、E2B/E4B、quant優先順位、公開面、recipe/core境界と同梱通知                           | 新規指摘なし。公式checkpointからの再変換・全byte監査と公開判断は今回行っていない       |
| 主担当                   | CLIのローカル選択、短文脈の履歴・KV再利用・中断、tokenizer、暖機とTTFT、PyTorchの重み復元と行lookup、品質の窓/採点、ブラウザの比較順・設定固定・保存・HTTPパス/Range | 新規指摘なし。添付の保存値は下記のとおり照合。新しい実ブラウザ生成・M2操作はしていない |

主担当はさらに`device.ts`のbatch最終決着と読戻し予約、`executor.ts`と`generation-context.ts`の予約・進行、
`pipeline-cache.ts`の共有Promise・拒否回収・scope区間、Gemma共通factoryとgreedy出力の呼出順、quant指定の優先順位を確認した。
PyTorchの`DevicePle`が通常PLE全表をGPUへ運ぶ可能性も検討したが、内部の`DiskPle`は全表のParameterを保持せず、行をCPUから読む実装であり、指摘にはしなかった。
比較基点より前のMTP実装や公開API変更、[既知問題](../known-issues.md)・[制約](../limitations.md)は新規不具合と区別した。

## 利用者のM2結果

添付`gemma-browser-2026-09-14T19-51-43.751Z.json`のSHA-256は
`b55056f813c24dfb0df6268af8653f8bef527fb9f8d4ea32c7c83782df6c3e96`。
[集計JSON](2026-09-14-m2-merge-review-results.json)に条件、各回の時間、前回の基準方式との比較を残す。
M2という機種名は利用者の報告に基づく。JSONはApple / metal-3、Chrome 153、fallback adapterではない。
通常E2B→QAT E2Bの2ロードで、各入力は初回1回・追加暖機1回・本計測3回、計20生成、失敗0。

容量128、chunk64、最大64生成、温度0、f32、quant `i4`と明示`parallel`、dense入力バケット、RMS融合、
RMS縮約workgroup、投入上限768。英語31 token・日本語37 tokenの固定入力である。
checkoutは`315732a`・clean。推論bundle SHAは
`b098c37dc4fa88c097d772a13a689db10533846e858a04a6708145736e8466e0`で、
[前回のM2追試](2026-09-13-m2-gemv-subgroup-adoption.md)と同じ。
前回の既存parallel各2ロードと、manifest SHA・全推論設定・入力・ブラウザ版・adapter欄・機能集合が一致した。
機能配列の列挙順は意味を持たないので集合として照合した。

以下は本計測3回の中央値。TTFTは最初のtokenまでの時間、decodeはその後の速度。
モデル取得・読み込み・文章復号は含まない。

| モデル  | 入力   | 暖機後decode | 暖機後TTFT |
| ------- | ------ | -----------: | ---------: |
| 通常E2B | 英語   | 30.772 tok/s | 344.225 ms |
| 通常E2B | 日本語 | 30.364 tok/s | 419.690 ms |
| QAT E2B | 英語   | 31.082 tok/s | 398.315 ms |
| QAT E2B | 日本語 | 31.967 tok/s | 487.415 ms |

入力×ロードの中央値をさらに集計すると、今回通常30.568 / QAT31.524 tok/s、前回の既存parallelは32.904 / 33.877 tok/s。
差はそれぞれ約−7.1% / −6.9%だが、別日時で今回各1ロード・前回各2ロードの記述的な比較に限る。
同じ推論bundleのため、この差を直近のコード変更による遅化とは判断しない。
電源・温度・クロック・他プロセスの負荷は記録されておらず、環境変動の原因は未確定。
QATの前回計測も入力×ロードの中央値が28.543〜34.961 tok/sに分布していた。

全20生成は64 token・停止tokenなし。初回・暖機・本計測を含め、前回parallelの40生成とモデル・入力ごとに
`tokenIds` / `stopToken` / `text`が一致した。画面の`deterministic`だけに依存せず保存値を比較した。
各回の`decodeTokensPerSecond = 63,000 / (elapsedMs - ttftMs)`も再計算して照合した。
中間値・logitsのu32一致、長文、E4B、広い品質、M2のGPU時間の帰属はこの添付では検証していない。
同じ20生成の再計測依頼や、比較画面の初期選択変更は不要。

## 検証の出所

添付のコピー・前回データ・集計スクリプト・全60生成の照合は
`outputs/bench/karume/2026-09-14_19-59-16_m2-merge-review-ndy3fu60/`に保存した。
既存実験や資産は上書きしていない。
文書準備の初回コマンドはPythonの引用符の構文エラーで書込み前に終了。
再準備は`outputs/bench/karume/2026-09-14_20-03-58_merge-review-record-mfhhbime/`へ保存した。
この準備も同じACTIVE_DESIGNのバックアップ名を2回作ろうとして排他的作成で停止した。結果本文・JSONと更新日のみ書込み済みと確認し、残る索引更新は新しい専用ディレクトリのスクリプトで完了した。失敗直後のfmtは実行されたが、ベンチや検証を旧候補で続行していない。

エージェントからの補助テスト報告は、主担当が保存ログを直接確認した検収件数には合算しない。
GPU担当とQAT担当の一部テストが実GPUを自動検出して実行された。主担当からはGPUテストを禁止しており、これは委譲時の手順違反である。
その間に主担当のGPUテスト・ベンチは実行していなかったが、担当間のGPU実行の重複は確認できない。
この補助結果をGPU検収の根拠には使わず、エージェント終了後に主担当が単独で全体verifyを実行した。

exporter/recipeの実装は今回変更していないため、直前の全体Python検証は[レビュー資料](2026-09-14-merge-review.md)を参照する。
主担当の計測ツールCPU検証は22 passed（2.64秒、exit 0）。ログは
`outputs/bench/karume/2026-09-14_20-07-47_merge-reviewed-cpu-gjw8_q5b/tests.log`。
品質の採点・PLEの行取得・固定SRQ・計時を含む既存テストを実行した。

コミット前の`deno task verify`は**2,959 passed（802 steps）/ 0 failed / 5 ignored（2 steps）**、
テスト26分3秒、verify全体1564.415秒、exit 0。
ログは`outputs/bench/karume/2026-09-14_20-07-17_merge-reviewed-verify-3sdefjle/verify.log`。
RTX 3080 Tiで実GPUテストを実行し、GPUベンチとは並走していない。
5 ignoredは母音検出の実音声4件と配布形1件の資産不足。2 ignored stepsはDenoのsubgroup機能不足。
この件数とは別に、BiRefNet HR / Lucida 2048の実画像golden不足で列挙されなかったケースがある。
全条件の実行済みを意味しない。今回のChrome実走やM2数値検証へ読み替えない。

文書のローカルリンク280箇所・集計JSONの一致と、検証開始時からTS/WGSL/HTML 782ファイルのSHA不変を確認した。
検証後の変更はこの検収記録と引き継ぎへの追記だけで、別途fmtと差分を確認する。
文書準備の残る索引更新は`outputs/bench/karume/2026-09-14_20-06-37_merge-review-record-retry-gmio681y/`に保存。

## 未完と次の作業

今回のレビューは未完の最適化や公開検収の完了を意味しない。残件は[backlog](../backlog.md)を正本とする。
M2のGPU費用帰属、RMS融合のモデル既定化、E4B・長文・広い品質、MiniCPM/Qwenの公開pipeline、QAT公開資産は引き続き未完。
subgroupとI4 L8の不採用判断を変更する新しい根拠は今回ない。

マージする際は対象ブランチのheadと今回検証したheadを再照合する。
今回のレビュー範囲と未検証条件を添えて判断できる状態までとし、マージ・push・公開は別操作として残す。
修正が必要な新規指摘がないため、実装の追加変更は行わない。
