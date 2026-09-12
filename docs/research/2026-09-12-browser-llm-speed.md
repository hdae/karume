> 2026-09-12時点の実測と調査。本文の表はRTX 3080 Ti / Linux / Chrome。利用者のM2実測は末尾のリンク先に区別して保存する。

# Gemma E2B / QAT E2B のブラウザ速度比較

## 目的と再現方法

利用者のtok/s比較を優先し、[Deno / PyTorch基準](2026-09-12-llm-speed-baseline.md)に続いて、
karumeとTransformers.jsを同じChromeで測る。M2でも`deno task bench:llm-browser`と
`http://localhost:8787`だけで起動できる[計測ページ](../../tools/llm-speed/browser/README.md)を追加した。
通常 / QAT E2Bを選び、初回、追加暖機1回、本計測3回を順に実行する。JSONは画面から保存できる。
両エンジンの公開pipelineを使い、runtimeのカーネルやモデルの既定動作は変更しない。

入力は前回の固定英語31 / 日本語37 tokenをそのまま[fixture](../../tools/llm-speed/browser/cases.json)へ移した。
温度0、最大64新規token、停止ID `[1,106,50]`。両エンジンへ同じ入力IDを渡す。
karumeではtokenizer / chat templateがfixtureと一致することも確認する。
TTFTは最初の非停止tokenまで、decode速度は`(配送数 - 1) / (完了 - 最初の配送)`。
デコード文字列生成・画面更新は計測外。0〜1tokenの速度はnullとする。

モデル読込は別計測。karumeはsequence作成を含めて生成を測り、終了後のsequence.disposeは計測外。
Transformers.jsは入力Tensorを準備してからgenerateを測る。generate内部の準備とKV解放は返却前なので含む。
独立iframeでモデルをロードし、エンジンを切り替える前にモデルとGPUを解放する。
反復ごとに会話・KVを作り直し、前の回答のキャッシュは再利用しない。
日本語の「初回」は英語計測後なので、プロセス最初の生成ではない。

## 実行条件とモデル差

ライブラリとONNX revisionの正本は[config.ts](../../tools/llm-speed/browser/config.ts)。
Transformers.js 4.2.0とONNX Runtime Web 1.29.0の公式配布ファイルを使う。
Transformers.jsのbrowser ESMは外部importするORT実行部とTensorクラスを同じORTモジュールへ解決する。
JS/WASMを同じ版に揃え、asyncify WASM、1 thread、WebGPU EPを使用する。
依存パッケージの実装は改変・転記していない。ロードされるsessionはembed_tokens / decoder_model_mergedの2本に限定する。

| 対象       | karume                                           | 公開ONNX                                                                        |
| ---------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| 通常E2B    | 既存のi4線形 / i8埋め込み・head・PLE、f32演算    | q4f16。242 MatMulNBits、2 GatherBlockQuantizedは4bit / block32                  |
| QAT E2B    | 固定INT2/4/8とSRQ、f32演算                       | q2f16。MatMulNBitsは2/4/8bit / block32の混在。埋め込みgatherは2/4bit / block256 |
| キャッシュ | capacity128 / chunk64、既定のPLEホストキャッシュ | exportされたattention graph、可変長KV                                           |

QAT ONNXのグラフにはSRQ専用opや明示的な丸めopを見つけていない。
QATという名前だけでkarumeの固定丸め契約と等価とは扱わない。通常版も量子化が異なり、重みの同一性は主張しない。
これは利用可能な配布物ごとの比較で、エンジンだけの性能差や品質の優劣を測る実験ではない。
公開配布物は[通常ONNX](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX)と
[QAT mobile ONNX](https://huggingface.co/onnx-community/gemma-4-E2B-it-qat-mobile-ONNX)。

## Linuxでの実行準備

headlessモードでは3種類の起動条件でWebGPU adapterがnullになったため、Xvfb上の**headed Chrome**をPuppeteerで自動操作した。
「headlessで成功」とは扱わない。ユーザー向けサーバーやM2の起動手順にはLinux用のフラグを持ち込まない。
Chrome 153 / 152とも通常設定ではRTXのshader-f16を公開しなかった。
Vulkan APIへの直接照会ではshaderFloat16 / shaderInt16 / storageBuffer16BitAccessが有効だった。
Dawnの[NVIDIA向けf16実験toggle](https://dawn.googlesource.com/dawn/+/5e9a4865b1635796ccc77dd30057f2b4002a1355%5E%21/)に基づき、
`--enable-dawn-features=vulkan_enable_f16_on_nvidia`を比較実験のChromeへ指定した。
数値適合の問題を背景に既定無効の機能であり、このフラグ付きRTX結果を通常M2の結果へ外挿しない。

この環境はフォントが無いためUI描画でブラウザのnavigationが失敗した。
実験ディレクトリにNoto Sans JPとOFLを保存し、専用fontconfigで解消した。システム設定は変更していない。
生データのflagsはPuppeteerが実際に起動したspawnargsを保存する。

## 結果

正本は[結果JSON](2026-09-12-browser-llm-speed-results.json)。各値は本計測3回の中央値。全て64tokenを配送した。

### ONNXをローカル配信

| モデル | engine       | 入力          | 初回TTFT ms | 暖機後TTFT ms | decode tok/s |  decode範囲 |
| ------ | ------------ | ------------- | ----------: | ------------: | -----------: | ----------: |
| normal | karume       | english-list  |       506.7 |         103.0 |        48.67 | 47.98–51.68 |
| normal | karume       | japanese-list |       175.0 |         130.5 |        50.49 | 50.15–50.92 |
| normal | transformers | english-list  |       445.2 |         284.8 |        12.41 |  5.39–13.34 |
| normal | transformers | japanese-list |       346.6 |         333.3 |        11.11 |  5.23–13.18 |
| qat    | karume       | english-list  |       289.8 |         107.2 |        54.12 | 51.26–54.14 |
| qat    | karume       | japanese-list |       234.6 |         134.8 |        52.49 | 51.42–53.61 |
| qat    | transformers | english-list  |       390.9 |         299.8 |        11.08 | 10.87–11.79 |
| qat    | transformers | japanese-list |       411.5 |         326.2 |        11.70 | 10.62–11.82 |

### ONNXをHugging Faceから取得（既定手順）

| モデル | engine       | 入力          | 初回TTFT ms | 暖機後TTFT ms | decode tok/s |  decode範囲 |
| ------ | ------------ | ------------- | ----------: | ------------: | -----------: | ----------: |
| normal | karume       | english-list  |       660.6 |         103.6 |        51.07 | 51.04–51.26 |
| normal | karume       | japanese-list |       172.8 |         127.3 |        50.67 | 50.28–51.05 |
| normal | transformers | english-list  |       552.8 |         271.0 |        11.71 |  5.06–11.76 |
| normal | transformers | japanese-list |       371.6 |         303.5 |        11.77 |  6.55–13.05 |
| qat    | karume       | english-list  |       490.9 |         105.4 |        52.78 | 50.47–53.03 |
| qat    | karume       | japanese-list |       233.5 |         134.1 |        52.61 | 52.00–53.35 |
| qat    | transformers | english-list  |       485.8 |         353.2 |        12.65 | 12.44–13.32 |
| qat    | transformers | japanese-list |       296.3 |         348.8 |        11.97 | 11.21–13.79 |

両取得経路を合わせて80生成で、各エンジン内の5反復のtoken列は一致した。通常/QATのkarumeは、前回Denoの英語/日本語の生成列とも一致した。
エンジン間の生成列は異なった。Transformers.jsには反復間の大きな速度差もあるため、中央値だけで安定した上限速度や原因を断定しない。
GPUベンチ同士やGPUテストは並走させていない。比較で固定したのはGPU、Chrome153、入力ID、停止条件、生成数、計測式。量子化と演算精度の差は残る。

## 生データと補助確認

専用ディレクトリ:
`outputs/bench/karume/2026-09-12_transformers-js-e2b-jSCqvm/`。

- `downloaded-files.json`: 公開ONNXの取得revision、サイズ、SHA256。LFS SHAとも照合。
- `browser-probes.json`: headlessのadapterなしという観測。
- `gpu-info-v3.txt` / `vulkan-features.json`: ChromeとVulkanの機能照会。
- `onnx-graphs.json`: 外部重みを展開せず検査したop / 量子化属性。
- `product-v3-karume-result.json`: Chrome152でのkarume予備計測。
- `product-v4-transformers-result.json`: Chrome153 / ORT1.29での予備計測。
- `product-ort126-transformers-result.json`: Transformers.js同梱ORT1.26 devでの通常版追試。
  実験用bundleの版指定だけを変更し、外部ライブラリ本体は変えていない。
  通常版は暖機後およそ12 tok/sで、ORT更新だけでは約10〜13 tok/sという傾向を説明できなかった。
  実験ページの環境見出しは製品configの1.29表示が残るが、結果JSONのonnxruntimeは実行版1.26 devを記録する。

前回のPyTorch基準は復元したdense演算または公式QAT層であり、専用CUDA量子化カーネルの上限ではない。
また利用者が紹介した[WebML Community Space](https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels)は
専用カーネル実装であり、ここで動かしたTransformers.js / ORTとは別の比較対象である。
広い品質評価は残る。利用者によるM2の同一ページでの実測は下記へ追記した。

## 検収

サーバーの範囲応答、範囲外拒否、HEAD、読取り中の中断、パス越境・symlink越境拒否を自動テストした。
計測ページの「JSONを保存」をPuppeteerから実際に押し、ダウンロードしたJSONと画面の結果オブジェクトの一致を確認した。
ログは`product-export.log`、保存物は同実験の`export-check/`。これは追加の10生成による画面操作検収で、上表の80生成とは別。
ブラウザと実験サーバーを停止し、全体検証は別の専用ディレクトリ`outputs/bench/karume/2026-09-12_browser-verify-9UMwHy/`で実施した。

`deno task verify`は**2,935 passed（762 steps）/ 0 failed / 5 ignored、25分13秒**で成功。
上記ディレクトリの`verify.log`が正本。GPUベンチとは並走させていない。
最終コードのbrowser bundleを再作成し、HF直接取得の4構成に保存したbundle SHAとの一致も確認した。
記録した生JSON2ファイルのSHA、karume/Denoの8組の生成列、文書の参照先も照合した。

## 利用者の M2 実測と改善の追試

通常版のみと、通常/QATを含む2回の提出を[生成列・全反復の計測値付きで保存](2026-09-12-m2-browser-speed-results.json)した。
再提出ではkarumeの暖機後decodeは通常/QATとも約19〜20 tok/sだった。Transformers.jsの通常版は2回の提出間で大きく変動しており、
一方の走行だけを安定した速度差と解釈しない。いずれも量子化・演算精度の条件差が残る。
[PLE行キャッシュの調査](2026-09-12-ple-row-cache.md)にM2の表と、その後のRTXでの費用分解・改善前後の比較を記録する。
改善後のM2は同じ `deno task bench:llm-browser` で再計測できる。提出済みのM2結果は改善前のコードによるもの。

## WebML Spaceとの比較とheadlessの追試

[WebMLの実測](2026-09-12-webml-browser-speed.md)で、Xvfbを起動してDISPLAYを渡すとheadlessでもRTXを取得できた。
以前のheadless失敗は起動条件に依存していた。Spaceの行列機能の制約を明示して実際の画面を自動操作し、
容量8,192・同一入力ID・生成数64/256/1,024・文字列復号込みでKarumeとも比較した。
短い生成でもWebMLの方が速く、生成長だけでは差を説明できない。数値契約・品質の同等性は未検収。
同じ記録に、PLE行キャッシュ適用後の利用者M2実測と40生成の一致も追加した。
