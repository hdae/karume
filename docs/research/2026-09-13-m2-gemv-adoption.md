# M2のGEMV並列加算とquant定義での採用

> 2026-09-13時点の利用者実機測定と採用判断のスナップショット。M2での操作は利用者が行い、こちらは保存JSONの検査と生成文の読解を行った。

## 実測

[集計JSON](2026-09-13-m2-gemv-adoption-results.json)に元ファイルの保存先・SHA256・各回の範囲を記録する。
提供ファイルは `gemma-browser-2026-09-13T05-00-09.585Z.json`。
commit `9e859ec` のclean tree、前回のRTX検収と同じbundle hashである。
Apple M2は利用者の申告、JSONのadapterはapple / metal-3、Chromeは153。

同じ入力・capacity128・chunk64・greedy・最大64 token。各条件は初回1回、追加暖機1回、計測3回。
計6構成×英語/日本語、60生成すべて64 tokenで、各条件内のtoken列・文章・停止結果は反復一致。
保存されたtok/sも `(tokens - 1) / (elapsed - TTFT)` から再計算して一致した。failuresは空。

| モデル・入力     | 逐次 tok/s | 並列 tok/s |  倍率 | TTFT中央値 逐次→並列 |
| ---------------- | ---------: | ---------: | ----: | -------------------: |
| 通常 E2B・英語   |     26.150 |     32.597 | 1.247 |     342.35→340.15 ms |
| 通常 E2B・日本語 |     25.945 |     32.659 | 1.259 |     634.10→634.98 ms |
| QAT E2B・英語    |     29.289 |     34.420 | 1.175 |     388.82→387.79 ms |
| QAT E2B・日本語  |     28.288 |     34.363 | 1.215 |     763.56→763.28 ms |

通常版は約25％、QATは約18〜21％改善。QATの計測3回の範囲は、逐次が英語26.86〜30.73 / 日本語26.55〜30.69、
並列が英語30.96〜34.42 / 日本語32.13〜34.38 tok/s。単一のUI順序でABBAの順序反転は無く、倍率を精密な定数とは扱わない。
別途行ったRTXの順序交替でも利得を確認済み（[前日の統合検収](2026-09-12-chrome-gemv-parallel.md)）。
この64 token・capacity128の値と、RTXの256 token・capacity8192の値を同条件として混ぜない。

同時測定のTransformers.jsは通常19.35 / 19.05、QAT19.94 / 19.97 tok/s（英語/日本語）。
これはONNX q4f16 / q2f16、Karumeは混成packed重みとf32なので、品質・演算の同等性は保証しない。
WebMLの専用カーネル版とは別の実装であり、この比較をWebMLへの到達として扱わない。

## 品質と採用判断

通常のA/Bは両言語で64 tokenが全一致。QATは英語の19番目、日本語の36番目から異なる。
保存文を読み、同じ質問に対する番号付きの助言、文意の連続性を確認した。
QAT英語は目標設定や言語への接触を勧める語句が変わり、日本語も表現が変わるが、
今回の短い出力には無関係な文章や繰り返しの崩壊は見られない。末尾は64 token制限で文の途中までである。

前日の固定ARC-Easy64問でQATは45→44正答（token数正規化44→43）、平均NLLは6.460015→6.500792、
perplexityは639.070→665.669。品質不変の証明ではないが、現時点の証拠では大幅劣化として棄却するほどではないと判断する。
通常版の小規模部分参照もほぼ一致した。詳細なデータ選択とCPU参照は前日の記録を正本とする。
長文・多様な指示・コード・広い日本語評価を完了したという意味ではない。

利用者は数値一致を比較検証のための経路として保持し、速度と品質で高速化の採否を判断する方針を明示した。
今回の速度利得・固定品質評価・保存された生成文を根拠に、通常/QAT E2Bの既定quantとして採用する。
利用者が広い品質を検収したとは記録しない。担当による採否判断は[ADR 0058](../decisions/0058-numerics-opt-in-contract.md)と
[ADR 0098](../decisions/0098-linear-gemv-parallel.md)へ反映する。

## quantと実行指定

- `i4`: 従来の重みと逐次GEMVを保持する参照用quant。
- `i4-gemvpar`: 同じ重み写像に `session: { linearGemvReduce: "parallel" }` を加える。
- 通常/QAT E2Bの新しい配布recipeは `defaultQuant: "i4-gemvpar"`。QAT E4Bは `i4` を維持する。
- 優先順位は呼び手の明示指定 → quantの宣言 → runtimeの逐次加算。
  `fromAssets`はquantを持たず、未指定時は逐次。モデルのサイズ・形・GPU名から加算方式を自動選択しない。
- カーネルと既存goldenは変更しない。MTPの同じ設定内のverify行0 / decode一致を引き続き検査する。

CLIは `--quant i4` / `--quant i4-gemvpar` と `--linear-gemv-reduce` の上書きを持つ。
Chromeは既定で「quant定義に従う」。同じquantで逐次/並列を指定して比較する口も残す。
出力JSONはquant・有効な加算方式・上書き有無を保持する。
既存のローカル配布形や公開pinは変更しない。新quantを使うには更新recipeで別の配布形を組み立てる。
古い配布形でも `--linear-gemv-reduce parallel` は使用できる。重みの再量子化は不要。

## 検証と次の調査

重点検証は23 passed（138 steps）/ 0 failed。
`outputs/bench/karume/2026-09-13_quant-focused-v16ydn1h/test.log`に保存した。
実重みの通常/QATで既定quant・参照quant・明示parallel・明示sequentialを走らせ、適用キーと出力の対応を確認。
未対応のquant.session欄は重み取得前のadmissionで拒否する。期待値や数値許容誤差を弱めていない。

配布recipeの検査はexporter **3,227 passed / 1 skipped**、recipes **2,800 passed / 4 skipped**。
exporterのログは `outputs/bench/karume/2026-09-13_quant-python-tests-s4q535dh/exporter.log`、
最終recipeコードの再検査は `outputs/bench/karume/2026-09-13_quant-python-final-y4vpsxul/pytest.log`。
後者は変更したPythonファイルのSHA256も保存し、走行後に一致を確認した。

実際の配布組立てと `verify_dist` も完了した。既存配布形との全資産のパス・サイズ・SHA256は一致する。

- 通常E2B: `outputs/bench/karume/2026-09-13_quant-distributions-nqqe9kt9/karume-gemma4`。
  `dist.py --pipeline gemma4 --series outputs/series --out <上記の新規パス>`。
- QAT E2B/E4B: `outputs/bench/karume/2026-09-13_quant-qat-family-7zrm_ov9/karume-gemma4-qat`。
  `dist.py --pipeline gemma4-qat --model e2b --model e4b --series outputs/bench/karume/2026-09-11_qat-integration/recipe-series --out <上記の新規パス>`。
  E2Bの既定だけが `i4-gemvpar`、E4Bは `i4`。

最初のQAT組立ては既定選択のE2Bのみで成功したが、比較スクリプトがE4Bもある元配布形とのモデル集合一致を要求して失敗した。
そのログは最初の専用ディレクトリへ保持し、上記の別ディレクトリで両モデルを明示して検査し直した。
既存配布形の変更・上書きはない。

Chrome 153 / RTXで、新しい配布形を使うUIの既定選択と逐次/並列A/Bを自動操作した。
計6構成・60生成は前日に検証した同じ加算指定のtoken列・文章・停止結果と一致。
既定の2構成は `quant=i4-gemvpar` / 有効な加算 `parallel` / 上書きなしとして保存された。
JSONダウンロードも表示データと一致する。画面画像は保存したが目視検査は行っていない。
生データは `outputs/bench/karume/2026-09-13_quant-browser-tnueg7ce/`。

Deno CLIは通常/QATで既定・明示parallel・参照quant・明示sequentialを比較し、通常版はMTPも追加した。
9起動で英語→日本語→reset→英語を自動入力し、既定/明示parallel、参照/明示sequential、通常版MTP/既定の文章が一致。
短いCLI確認から速度の結論は出さない。生データは `outputs/bench/karume/2026-09-13_quant-cli-61jbapv8/`。

全体検証は **2,953 passed（784 steps）/ 0 failed / 5 ignored、25分34秒**。
`outputs/bench/karume/2026-09-13_quant-verify-0mwfzbcq/verify.log` が正本。
検証開始時から追跡対象の全ファイルのSHA256が同じことを確認した。結果追記後はfmt・リンク・差分を再確認する。
最後に作り直したbrowser bundleも、上記UI検収で使ったbundleとバイト一致する（同ディレクトリの `bundle-verification.json`）。
次は入力バケットH-23の比較を行う。既定・少数追加・8行ごとの追加を、通常/QAT・容量128/8192・入力長10種で往復比較する。
準備は `outputs/bench/karume/2026-09-13_prefill-bucket-survey-_s_ei117/PLAN.md`、台本は同ディレクトリの `run.py`。
この記録のコミットと全体検証が完了してからGPUを使い、結果は専用ディレクトリへ排他的に保存する。
製品の既定バケットはまだ変えない。埋め込み取得のGPU内処理、先行投入、RMS/SRQ/linear周辺の融合も残る。
