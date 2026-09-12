> 2026-09-12時点の実測。WebML/Karumeの比較はRTX 3080 Ti、M2の追試は利用者の提出JSONによる。

# WebML Space の headless 実行と Karume の生成長比較

## 結果

同じChrome・GPU・入力ID・会話容量・最大生成数で比較したところ、WebML版は64tokenでも高速だった。
1,024tokenまで伸ばすことで表示速度が上がる、という仮説はこのRTX実測では支持されなかった。
Karume QAT E2Bに対して約4.6〜5.0倍の差がある。ただし、重み表現・丸め契約・生成列が同一の比較ではない。
M2でこの倍率になるとは主張しない。

正本は[実測JSON](2026-09-12-webml-browser-speed-results.json)。各条件3回の中央値。
TTFTは最初の非停止tokenまで、速度は `(生成数−1) / (完了時刻−最初の配送時刻)`。

| 最大生成数 | WebML QAT E2B tok/s | Karume QAT E2B tok/s | Karume 通常E2B tok/s |
| ---------: | ------------------: | -------------------: | -------------------: |
|         64 |              285.33 |                57.37 |                52.57 |
|        256 |              280.25 |                58.26 |                52.87 |
|      1,024 |              262.62 |                57.72 |                52.65 |

| 最大生成数 | WebML TTFT ms | Karume QAT TTFT ms | Karume 通常TTFT ms |
| ---------: | ------------: | -----------------: | -----------------: |
|         64 |         102.0 |              118.7 |              114.5 |
|        256 |         102.0 |               58.5 |               70.2 |
|      1,024 |         101.8 |               59.7 |               69.4 |

TTFTには前の実行によるPLE行の保持状態なども影響する。最大生成数を大きくするとTTFTが短くなる、という因果関係にはしない。
Karumeの64tokenは最初の実行や長い生成の直後も含み、都合の良い反復を選んでいない。

WebMLの1,024token走行について、配送時刻間の区間速度も算出した。3回の中央値は、
最初のtokenから64番目まで283.27、64→256番目277.22、256→512番目261.76、512→1,024番目255.62 tok/s。
後半の速度低下が文脈長・文字列復号・GPU実行のどれに起因するかは、ここでは帰属していない。

## 揃えた条件と残る差

- RTX 3080 Ti、Linux、Chrome 153.0.8010.36、Puppeteer、`headless: true`。
  Xvfbの `DISPLAY=:98` を両者で使用。GPUベンチ・GPUテストは並走していない。
- 会話容量8,192、温度0、投機デコードなし。WebMLの容量は実際のcacheから取得した。
- 同じ英文で100件の助言を求める。Karume tokenizerのchat入力IDがWebMLの41tokenと一致することを、生成前に検査した。
  文面とIDはJSONに保存している。
- 最大生成数の順序は `64,1024,256,256,1024,64,64,256,1024`。全て指定数まで生成した。
  各実装内で1,024tokenの生成列と短い生成の接頭辞が一致し、同じ長さの3反復も一致した。
- WebMLの既定暖機に合わせ、Helloを6/8token生成した後、固定英語・日本語を各1回生成してから長さ比較へ進む。
  この2つの事前生成はモデルごとに停止までの生成数が異なる。前の回答のKVを本計測へ持ち越さない。
- 両者とも各tokenで全生成列を文字列へ復号し、その後の配送時刻を採った。長さ比較では画面描画をしない。
  WebMLは公開 `generate()`、Karumeは公開 `sequence.generate()` とtokenizerを使用する。
  token化は計測内、モデル読込みは計測外。
  Karume標準chatの逐次復号も別走行で確認した。結果は次節に示す。

残る差は次のとおり。

- WebMLは[公式QAT mobile](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers)を専用実装で読む。
  Karumeは変換済みの通常i4/i8、または固定INT2/4/8とSRQ（静的な再量子化）を既定のf32演算で実行する。
  WebMLはf16/subgroupsを有効にしたdeviceを使うが、全演算の精度・丸め契約や資産のbyte一致は検収していない。
  長さ比較の生成列は両KarumeモデルともWebMLと18token目から異なる。品質低下を立証したわけではない。
- WebMLは既存KVをresetし、Karumeはsequenceを各回作り直す。Karumeのsequence確保・破棄も時計の外に置いた。
  公開APIの寿命管理が異なるため、完全に同一の処理ではない。
- WebMLはprefillの32/128/256行ブロック、Karumeはchunk64と既定バケットを使う。
  最大生成数の比較のためにカーネルやバケットの既定は変更していない。
- WebMLの重みはHFから取得し、Karumeは既存ローカル配布物をHTTPで供給する。
  KarumeのPLEは既定の全量2shard予算と256行キャッシュ。ホスト読取りも含む実装全体の比較である。
- WebMLの測定後にKarume QAT、通常版の順で測った。エンジンを交互にしたABBAではない。

## Karume標準chatでの追加測定

普段の公開APIでも確認するため、同じモデル・41入力token・容量・暖機・長さの順序で
`pipeline.chat`を各3回測った。逐次復号を使い、sequenceの確保と破棄も時計に含む。
`onToken`の時刻は復号前、完了時刻は最後の文字列配送と後始末の後である。

| 最大生成数 | QAT tok/s | 通常版 tok/s | QAT TTFT ms | 通常版 TTFT ms |
| ---------: | --------: | -----------: | ----------: | -------------: |
|         64 |     56.26 |        51.49 |       125.6 |          117.5 |
|        256 |     57.11 |        52.76 |        65.8 |           82.5 |
|      1,024 |     56.70 |        52.50 |        62.5 |           73.2 |

18走行全てが全列復号の比較とtoken列・文章とも一致した。標準chatでも速度は近く、
復号方式だけでWebMLとの差を説明できない。実行順を交替したAPI間A/Bではないので、
小さな速度差を逐次復号の損益とは断定しない。数値はJSONの`karumeNativeChat`。

## 公開Spaceの画面操作

[Space](https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels)へ直接アクセスし、ロードボタン・送信ボタンをPuppeteerで操作した。
実際のUI表示は、headlessで英語450token / TTFT52ms / 272.6 tok/s、日本語750token / TTFT103ms / 266.5 tok/s。
スクリーンショットと全tokenの配送時刻を保存した。UI操作の2生成は、上の生成長比較とは別枠である。

[画面の実装](https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels/blob/158f16ae0f672943ca304d59c47c8e3a264e399e/index.html)は最大4,096token、
ロード後に短い暖機を行い、tok/sからTTFTを除く。このため、長い生成で単に初回応答時間が薄まる指標ではない。
画面の最終値には文字列復号・逐次描画も含まれる。

先にXvfb上の通常Chromeでも同じ比較を行った。生成長64/256/1,024で285.20/281.30/263.12 tok/s。
通常Chromeとheadlessの11組（UI2件＋長さ比較9件）で生成列が一致した。

## Linuxの起動条件と失敗の切り分け

1. Xサーバーなしのheadlessではadapterがnull。Xvfbを起動して `DISPLAY=:98` を渡すと、headlessでもRTXを取得できた。
   [前の比較](2026-09-12-browser-llm-speed.md#linuxでの実行準備)の「headlessでは取得できなかった」は当時の起動条件に限る。
2. `--enable-gpu`、Vulkan/ANGLE、blocklist解除、NVIDIAのf16実験toggleなどは、前のRTX比較と同じ。
   実際の全spawnargsをJSONに保存した。M2向けの起動手順にこのLinux用設定を足す必要はない。
3. Spaceの既定ロードは `DenseGemvSgmat` の行列形式 `M(8), N(0), K(8), f16` が未対応として暖機中に失敗した。
   Spaceの公開ロード引数 `runtimeOptions.disabledFeatures` で `chromium-experimental-subgroup-matrix` だけを無効化した。
   この設定差は上の全WebML速度に適用する。ソースやカーネルを書き換えていない。
4. Karumeの最初の台本はSession構築中に `Instance dropped in popErrorScope` で失敗し、生成値を作らなかった。
   `navigator.gpu` とadapterをページ上に保持する台本に変えると全測定が完了した。
   成功走行のログにも同種の警告が記録されており、保持不足が根因とまでは断定しない。
   [Dawnの実装](https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/wire/client/Device.cpp)でも、この警告はshutdown時のcallback中止に使われる。
   製品runtimeの修正はしておらず、原因の最小再現は残件。

Spaceの既存参照リポは `~/workspace/reference/gemma-4-webgpu-kernels`。SpaceのSHAと配信JSのSHA256を保存し、既存参照とbyte一致を確認した。
配信HTMLだけはHFのcreator metadata挿入がある。モデルはSpaceが `main` を解決しており、APIで確認したSHAを記録したが、モデル全byteの検証済みpinとは扱わない。
Space rootにLICENSEの明示を見つけていないため、今回は閲覧と公開APIの実行に限り、第三者の実装をKarumeへ複製していない。

## M2のPLE行キャッシュ導入後の追試

利用者の `gemma-browser-2026-09-12T17-37-30.505Z.json` はcleanな `a0d6d13`。前回と資産・設定が一致し、40生成全てが前回の同条件のtoken列・停止結果と一致した。
元ファイルのSHA、比較の中央値、保存先は実測JSONの `m2Followup` にある。

| モデル・入力 | Karume tok/s 前回→今回 | TTFT ms 前回→今回 |
| ------------ | ---------------------: | ----------------: |
| 通常・英語   |          20.02 → 27.51 |     451.9 → 341.6 |
| 通常・日本語 |          19.08 → 27.84 |     734.1 → 639.6 |
| QAT・英語    |          20.12 → 29.37 |     520.8 → 393.1 |
| QAT・日本語  |          20.21 → 29.81 |     875.2 → 772.9 |

Transformers.jsも約9/10から19/20 tok/sへ戻ったので、前回との差を全て行キャッシュの効果とは断定しない。
通常版だけの最初の提出との比較では、Karumeは約27〜30%上がり、Transformers.jsは約−0.4〜+1.2%だった。
同じ文章の反復であり、一般入力に対するM2の改善幅や、QATのM2/RTX生成列差の原因は未検収。

## 保存先と次の調査

- `outputs/bench/karume/2026-09-12_webml-browser-length-rqri7fkd/`: Spaceの来歴、起動probe、既定の行列機能での失敗。
- `outputs/bench/karume/2026-09-12_webml-no-matrix-length-heu8cz8g/`: Xvfb上の通常Chrome。
- `outputs/bench/karume/2026-09-12_webml-headless-x-display-qibm7pgr/`: Xvfbありheadlessでのadapter確認。
- `outputs/bench/karume/2026-09-12_webml-headless-length-oxkwgv7f/`: headlessのUI・生成長比較、スクリーンショット、集計台本。
- `outputs/bench/karume/2026-09-12_karume-webml-matched-w2alts5m/`: Karumeの実験コード・bundle・HTTPサーバー、最初のロード失敗。
- `outputs/bench/karume/2026-09-12_karume-webml-retained-gpu-du42cb5l/`: GPU取得元を保持したKarume比較。
- `outputs/bench/karume/2026-09-12_webml-karume-report-i990oz_0/`: 両実装の集計と検証。
- `outputs/bench/karume/2026-09-12_karume-webml-native-chat-cxjs80_f/`: 標準chatの追試。

次はこの差をGPU処理・ホスト処理・GPU/CPU間の待ち時間へ帰属する。WebMLのGPU上の埋め込み参照や複数stepの先行投入は候補だが、今回の実測だけで寄与率を断定しない。
広い品質比較、M2で同じ長さの比較、Karumeの入力バケットH-23、GPU Instance消失の最小再現は残る。

## 検証

製品コードは変更していない。`deno task verify`は2,936件成功（771ステップ）、失敗0、ignore5、25分6秒。
ログは`outputs/bench/karume/2026-09-12_webml-comparison-verify-5bo1s1m6/verify.log`。
その後の標準chat追記では、18組のID・文章・速度式・raw SHAを再検査し、文書のfmtと差分検査を実施した。
