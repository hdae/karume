# Chrome の量子化 GEMV 並列化と残る費用

> 2026-09-12 時点の隔離実験・統合検収のスナップショット。RTX 3080 Ti / Chrome 153 の値であり、M2 の実測ではない。

[先行追試](2026-09-12-chrome-gemv-followup.md)の大語彙INT8 c16を基準に、K方向の分担、重み配置、SRQを比較した。
[保存結果](2026-09-12-chrome-gemv-parallel-results.json)に集計と元ファイルのhashを置く。
製品の既定は維持し、K並列を`linearGemvReduce: "parallel"`として任意指定できるようにする（[ADR 0098](../decisions/0098-linear-gemv-parallel.md)）。

## 条件と比較の境界

既存の変換済みGemma通常/QAT E2B、capacity 8192、chunk 64、温度0、非投機の`pipeline.chat`。
[WebML比較](2026-09-12-webml-browser-speed.md)と同じ41入力IDから最大256 tokenを生成する。
各A/Bページで256 tokenを3回暖機し、ABBAを4組。モデルロード・暖機・TTFTはdecode速度に含めない。
GPUベンチ・GPUテストは逐次実行した。ChromeはXvfb併用のheadless/Vulkanで、起動flagsは保存台本に残す。

WebMLの公開JSを読んでK方向の分担・融合・先行投入を調査したが、ソースコードは複製していない。
今回はKarume自身の16 byte圧縮語の展開を共有し、workgroup内の固定順の木縮約を実装した。
subgroupの幅や跨workgroupの同期は仮定しない。

## K並列の適用範囲

実際に現れる26形状で、Kを2/4/8/16/32 laneへ配り、128 threadのworkgroupを複数出力へ割り当てた。
単体は暖機用GPU処理、64 dispatch/pass、6回の順序交替で比較。FP64参照は各形状・3入力分布の128出力行で調べた。
形状ごとに10％以上の利得があった25形だけを選び、I2の巨大語彙headは対象外とした。
微小な合成入力の最良値だけで既定を替えず、以下のモデル全体比較と数値検査も行った。

- INT4の長いKでは最大約5.4倍、通常版のINT8語彙headはc16基準で約1.78倍。
- INT2のup projectionは2 lane、down projectionは32 laneが有利だった。一律32 laneは遅い形を含む。
- 重み・scale・活性の精度は保持するが、部分和の加算順が異なるため既定とのbit一致は主張しない。

| 隔離候補     | QATの全体倍率中央値 | 通常版の全体倍率中央値 | 判断                                        |
| ------------ | ------------------: | ---------------------: | ------------------------------------------- |
| M=1だけ      |              1.3616 |                 1.5171 | MTPのverify行0とのu32一致を壊す             |
| GEMVの全M≤64 |              1.3878 |                 1.5621 | 通常版のTTFTが約70〜80→220 ms               |
| M=1..8だけ   |              1.3817 |                 1.5830 | 少数行検証とdecodeを揃え、長いprefillは保持 |

M=1..8版のABBA組平均はQAT A64.39〜64.63 / B88.08〜89.66 tok/s、通常 A59.22〜60.04 / B92.25〜97.77 tok/s。
通常の16走行は全て同じ256 ID、QATは各設定内で反復一致するがA/Bが18番目のtokenから分岐する。
WebMLの約280 tok/sには未到達。加算・再量子化・モデル表現の同等性も保証していない。

M=1だけのMTP検査では26万語彙の約24.9万要素で下位bitが異なった。
同じK分担をM=4/8にも使うと、既存のMTPテスト3件・10 stepsが全て成功した。
製品版ではattentionの逐次/並列の両方で、verify先頭行とdecode全262144値のu32一致を追加検査する。

### 熱と再現性

連続計測中にGPU温度92〜93℃とsoftware thermal slowdownを観測し、クロックも1485〜1905 MHzで変動した。
`2026-09-12_chrome-thermal-trace-phqkmtdg/samples.jsonl`に21:40 UTC以降の観測を保存。
記録が無い時間帯の熱状態は推測しない。最大tok/sだけを採らず、短いABBAと各組の倍率・範囲を残す。
冷却設定・クロック制御は変更していない。後半の速度下降を全てコードへ帰属しない。

## QATの品質差

固定入力から正解tokenを強制してlogitsを採点した。採点経路はcapacity 256、全logitsをCPUへ読み戻し、
JSのf64 logsumexpでNLL（負の対数尤度）を計算する。これは速度計測の経路とは別である。
最初の部分評価・格納型別評価のmetadataには生成用probeから継承したcapacity 8192 / greedyの欄が残っていた。
その欄は採点経路を表さない。後続の全小規模suiteでは修正し、実際の256/teacher-forcingを記録した。

ARC-Easyの固定64問、WikiTextの先頭8192文字から幅128・stride64の29窓、1870採点tokenを比較した。
データ選択・出典・license・token列は保存suiteを正本とする。標準leaderboardの全体スコアではない。

| QAT E2B / M=1..8              |     逐次 |     並列 |
| ----------------------------- | -------: | -------: |
| ARC正答                       |    45/64 |    44/64 |
| 選択肢token数で正規化した正答 |    44/64 |    43/64 |
| 平均NLL（低いほど良い）       | 6.460015 | 6.500792 |
| perplexity（低いほど良い）    |  639.070 |  665.669 |

小さい範囲ではNLL約0.63％、perplexity約4.16％の増加。QATの品質不変や改善を結論しない。
最初の8問・4窓319 tokenでは、通常版のNLLは逐次8.226834 / 並列8.226833と近かった。
QATは5.726092 / 5.787672。INT2だけを変えた候補5.754806、INT4/8だけ5.773892で、双方が差に寄与する。

同じ保存重みをPyTorch/TransformersのCPU float32・eager・4 threadで採点した共通8問/4窓は、
通常NLL8.226802 / ARC5問、QAT5.819148 / ARC6問。Karumeの同部分のARCは通常A/Bとも5問、QAT A/Bとも6問。
QATの並列版は平均NLLがPyTorchへ近づくが、tokenごとの差の平均絶対値は少し増えるため、全般的に良いとは言わない。
PyTorch側はf32 cross entropyで、採点末尾の演算精度もJSと同一ではない。
CPU品質計測の時間はGPU生成速度として扱わない。

## 並列化後のプロファイル

外部timestampを付けても通常のgreedy処理・passまとめを保つ計測と、1 dispatchずつ分ける順位用計測を分離した。
前者でprefillを除き31 decodeを集計すると:

| モデル / モード | GPU ms/token | 壁時計 ms/token | dispatch/token | pass/token |
| --------------- | -----------: | --------------: | -------------: | ---------: |
| QAT逐次         |       11.842 |          15.590 |           1617 |          3 |
| QAT並列         |        7.866 |          11.206 |           1617 |          3 |
| 通常逐次        |       12.702 |          16.452 |           1130 |          3 |
| 通常並列        |        6.780 |          10.194 |           1130 |          3 |

dispatch別の順位では、QATはSRQ487回、RMS norm242回、INT2/4 GEMVが残る。
通常版もINT4 GEMVが依然最大で、次がRMS norm。dispatchを分割した時間を上表のGPU時間へ足し合わせてはいけない。
壁時計とGPU時間の差約3.3〜3.4 msも残り、GPU常駐PLEと先行投入・小処理の融合が次の候補。
過去の観測ではSRQの直前がRMSであるものは70箇所で、RMS全242箇所を融合可能として数えない。

## その他の採否

SRQの固定境界探索を分岐展開し、uniform/storageの表を比較した。
10種のscale・境界と隣接値・符号・NaN/Inf等の71920値でu32一致。CPU側の探索比較394668件も一致した。
uniform展開の単体は小形で約1.10倍だったが、QAT全体4組の中央値は1.0059で安定した利得が無く不採用。
storage表は小形で逆行し、全体実行に進めなかった。既存SRQの固定丸め規則は変えていない。

INT4/INT8重み語の転置コピーは、通常版が全体1.0948倍、QAT1.0070倍、32走行の出力が一致。
追加コピーは通常1,341,063,168 byte（1278.94 MiB）、QAT379,060,224 byte（361.5 MiB）。
原本がprefill/embeddingに必要なので二重保持となる。費用に対する利得が小さく、今回の統合は見送る。
この隔離版はrecipe導出中にGPU前処理を入れた便宜的な実験であり、製品の導出相の不変条件を満たす設計ではない。

## 統合の検収

- 既定のWGSL snapshotは全て無変更で一致。並列版の新snapshotを3本追加。
- 全lane数、INT2/4/8、短いK・余るlane・端の列・group 32/512/2048/4096とM1/4/8を実GPUで検査。
  既存linearの許容誤差 `2e-4 + 4e-6 × |参照|` の内側でFP64参照とA/Bを検査し、帯を広げていない。
- 実際のSessionで6形状を通し、既定・明示指定・M>8の診断キーとM1/4/8のu32を検査。
- 構築時に不正な綴りとf16/a8演算との未実装組合せを拒否する。追加メモリ見積りは不要で、重み・storage bufferは増えない。
- 初回の重点検証の1失敗は、新規census用fixtureのscale形を`[N]`としていたため。
  containerのkeepdim契約`[N,1]`へ直し、単独再実行は1件/1 step成功。既存アサーション・数値条件は無変更。

### 統合版の再測定

`2026-09-12_parallel-integrated-speed-85xatug9`で通常の公開オプションを使って再度4組のABBAを実行した。
全32走行のtoken列・文章・停止結果は、対応する隔離版の設定と一致した。各設定内でも反復一致。

| モデル  | 既定 tok/s中央値 | 並列 tok/s中央値 | ABBA組ごとの倍率中央値 | TTFT中央値 既定→並列 |
| ------- | ---------------: | ---------------: | ---------------------: | -------------------: |
| QAT E2B |            64.29 |            89.13 |                 1.3867 |         66.4→64.5 ms |
| 通常E2B |            59.40 |            92.14 |                 1.5449 |         76.9→72.9 ms |

範囲はQAT並列83.06〜89.75、通常並列84.21〜98.47 tok/s。出力長は全て256。
隔離版より後半の速度が低いが、4組すべてで利得があり、最大値だけを根拠にしない。
統合版の新WGSLと実験版は生成する字面が違うため、数値の根拠はこの出力比較・FP64門・MTP門で持つ。

### Chrome画面とCLI

`2026-09-12_parallel-ui-retained-j1dhe9sf`で、通常/QAT × 既定/並列 × 英語/日本語を画面から実行した。
各入力は初回1回・暖機1回・計測3回で計40生成。各設定内のtoken反復一致、4構成の選択、JSONダウンロードと表示元データの一致を確認した。
この画面はcapacity 128・最大64 tokenなので、上のcapacity 8192・256 tokenの速度表とは分けて読む。
`deno task bench:llm-browser`を起動し、「比較対象: karume」「Karumeの行列計算: 既定と並列を比較」で追試できる。

最初のUI自動試行は生成前のadapter取得で失敗し、ログにexternal Instance消失を観測した。
`parallel-browser-ui-gf8za9pj`と`parallel-ui-diagnose-uzl4bwfx`に失敗・画面状態を保持。
以前のLinux Chrome probeと同じく、自動検証で各documentの`navigator.gpu`参照を保持した後は成功した。
この回避がChrome内部の根因を証明するわけではない。製品ページへ環境依存の回避処理は追加していない。

`2026-09-12_parallel-cli-by938wu4`でDenoの通常/QAT/通常MTPを逐次実行し、各3ターン・reset・正常終了・並列指定の表示を確認した。
速度の数値にはこの短いCLI動作確認を使わない。

全体検証は **2,950 passed（780 steps）/ 0 failed / 5 ignored、25分0秒**。
`outputs/bench/karume/2026-09-12_parallel-verify-final-x43w55vk/verify.log`が正本。
開始時と終了時で全差分ファイルのhashを照合し、検証中の変更が無いことも確認した。
直前の全体走行は、最終レビューで新familyをopbenchのlinear集計へ対応付ける必要が分かり、途中で明示中断した。
`parallel-verify-ff_sru_x/interrupted.json`に理由を保存し、対応とCPU検査7件を追加して上記の全体走行を最初から実行した。
その中断をテスト成功や数値失敗として数えない。

最終ソースから再生成したブラウザbundleはUI検証時の4構成のbundleとhash一致。
形状選択の25件も隔離版と一致した。`parallel-integration-report-h8h_xatw/parity.json`に記録した。
M2の実行と人による品質判断は残っている。

## 保存先と再開

すべて`outputs/bench/karume/`下。主要集計は冒頭のJSONから原本へ辿れる。

- `2026-09-12_chrome-vector-census-fixed-2aqozczi`: 26形状の単体掃引
- `2026-09-12_chrome-vector-smallrows-odyqtrzv`: M1..8のABBA
- `2026-09-12_chrome-vector-smallrows-quality-o8u9b3th`: 固定64問/29窓のQAT比較
- `2026-09-12_vector-quality-torch-sn3w3nbl`: 同保存重みのCPU部分参照
- `2026-09-12_chrome-vector-profile-qetz9d8w`: 並列化後のpass/dispatchプロファイル
- `2026-09-12_parallel-focused-m78is_f0` / `2026-09-12_parallel-census-recheck-p3lzm9l7`: 重点検証とfixture修正後の単独再実行

失敗した準備も保持した。`chrome-srq-table-osgwzv14`はTS型、`chrome-vector-census-o7ig_6y2`は分割headのshape解決、
`chrome-vector-selected-3bqywde_`は置換元の字面assert、`vector-mtp-m1-dos2mb6u`はimport-mapの不足で、GPU数値の失敗とは異なる。
後続のM1-only MTP数値失敗は`vector-mtp-m1-mapped-fsmjl6b9`。テストの門を緩めず、同じ加算形を少数行へ適用して解消した。
未実行の候補（storage表の全体版、旧packed全体版、全M版の広い品質評価）は計測済みとして数えない。

次はM2の明示指定A/Bと生成品質、その後にGPU常駐PLE・投入回数・RMS/SRQ/linear周辺の融合。
残る費用を再測定しながら進め、WebMLと同じ速度になったとは主張しない。
