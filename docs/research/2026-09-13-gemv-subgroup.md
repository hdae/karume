> 2026-09-13時点のM2追試・RTXプロファイル・GEMV候補の実測スナップショット。未検証GPU・長文・品質へ外挿しない。

# M2のRMS追試と並列GEMVのsubgroup化

## 判断

M2のRMS subgroup32は従来34.899→35.081 tok/s、約0.52%の差だった。反復の幅と重なり、TTFTもほぼ不変。
既定採用を裏付ける利得とは判断せず、ADR 0100の任意指定を維持する。
次のGEMV候補はparallelの入力配分・加算木を保ったまま、workgroup内の共有メモリと同期を32レーン内の値交換へ置き換える。
RTXではQATに全体の利得が再現したため、[ADR 0101](../decisions/0101-linear-gemv-subgroup.md)の任意指定として統合する。
通常版のRTX利得は追試で再現しなかった。[その後のM2追試](2026-09-13-m2-gemv-subgroup-adoption.md)でも高速化せず、
M2向けの既定採用を見送った。既定quant・保存形式は変更しない。

数値の集計は[保存JSON](2026-09-13-gemv-subgroup-results.json)。以下の中央値は信頼区間ではない。

## 利用者のM2結果

添付は`gemma-browser-2026-09-13T15-41-09.903Z.json`。
SHA-256は`0a59b894fbed4477ae0f49085186a75473111b851f5767a216b35169147e482e`。
checkoutは`744cdbe`・clean、bundleは`e8ef44d661dd69774ca08d16287376a17692fb9c4c05ec57ed9e423c65e5c70a`で前回のRTX UI検収と同じ。
QAT E2B・Chrome 153・parallel・dense・容量128・chunk64・最大64生成、RMS融合・投入上限768を固定した4設定40生成。

| RMS方式    | 暖機後decode |   英語TTFT | 日本語TTFT |
| ---------- | -----------: | ---------: | ---------: |
| workgroup  | 34.899 tok/s | 387.303 ms | 478.392 ms |
| subgroup32 | 35.081 tok/s | 387.205 ms | 478.265 ms |

英日それぞれの本計測3回中央値を、2ロード分まとめた中央値。decodeは英日×2ロードの4中央値を集計する。
workgroupの4値は34.393〜34.963、subgroup32は34.050〜35.408。範囲が重なり、安定した加速とは扱わない。
同一方式内のtoken/stop/text不一致は0。英語・日本語とも方式間で文が変わるが、短い回答に崩壊は見られない。
この40生成だけで広い品質や速度の同等性を証明したとはしない。

## 現在の費用内訳

RMSはworkgroupへ戻し、parallel・dense・融合・投入768を基準にRTX 3080 Ti / Chrome 153で再測定した。
製品`loadKarume`と同じ通常GPU greedy経路を使う。外付けtimestamp計器が要求した機能で診断経路へ変わらないよう、
実験コピーの`gpuTimingEnabled`だけfalseへ固定し、その変更をprovenanceへ記録した。
英日×通常/QATの各条件で3暖機、無計測→pass計測→無計測→dispatch分割計測→無計測、API計測、CPU samplingを各1生成。
計40生成のtoken/stop/textは同条件内ですべて一致した。警告・pageerrorは0。

| モデル  | 英語の通常pass計測decode | 英語のGPU時間/token | 日本語のGPU時間/token | decode dispatch/token |
| ------- | -----------------------: | ------------------: | --------------------: | --------------------: |
| 通常E2B |            109.852 tok/s |            6.429 ms |              6.484 ms |                 1,024 |
| QAT E2B |             99.905 tok/s |            7.465 ms |              7.515 ms |                 1,511 |

native passのprefill先頭2本（通常）/3本（QAT）を除き、63回のdecode×3 passのtimestampを集計した。
1トークンのwall時間は通常約9.1 ms、QAT約10.0 ms。GPU時間との差にはCPU処理・待機・投入境界などがあり、
単純に全差分をTypeScriptの実行時間とは呼ばない。CPU samplingの大半はidle/programであり、programの全量をJS演算と断定しない。

dispatchを個別passへ分けた帰属計測では、QATは並列GEMVとSRQ（固定量子化精度への丸め）が上位。
この計器ではQATのdecodeが約50 tok/sまで低下する。個別dispatch時間の総和を通常生成のGPU内訳率として扱わない。
SRQ探索短縮・RMS→SRQ・linear→SRQは既に全体比較まで終えて利得が小さかったため、まずGEMVの同期を対象にした。

## 候補と単体検査

既存parallelは1出力の圧縮語をL=2/4/8/16/32レーンへ巡回配分し、各レーンの部分和を共有メモリへ置く。
候補も語内の積和・scale・biasを維持し、幅L/2→1の各段で`subgroupShuffleXor`から相手の部分和を得る。
下位レーンだけを更新し、元の木の加算順を保つ。固定32レーンを要求し、subgroup内IDとsubgroup IDで出力列を定義する。
local IDとの配置対応を仮定せず、端の列も全レーンが値交換を通る。追加の重みコピーは無い。

- 短K・余るレーン・端の列N36・I2/I4/I8・I4 group32/512/2048/4096・M1/4/8を検査。
- 実形状25件を加えた205件・365,488出力で、既存parallelとのu32不一致0。
- 小さい形状は既存FP64許容帯も通過。実形状は全出力のu32比較を行い、巨大なFP64参照だけは作っていない。
- 単体速度は同一passで32 dispatchのtimestamp、各形状4暖機後ABBA×4。既存lane数の25形状は約1.001〜1.103倍。
  例えばI4 g512 N256 K1536は3.424→3.104 µs、I2 N1536 K12288は34.976→34.016 µs。
  単体の速度比を、そのままモデル全体の利得へ足し合わせない。

追加で全レーンの加算を無条件にした蝶型の値交換と、25形状×5レーン数を探索した。
1形状（I8 N262144・L32）は65,536 workgroupsとなりGPU上限65,535を超えるため無効。探索ハーネスの門漏れを修正し、
有効124形状と180数値検査を再実行した。304件・1,452,976出力のu32不一致0。
この探索では形状間で速度の変動もあり、別走行の絶対時間から枝削除や幾何変更の利得を確定しない。
大きなI4行列のレーン数変更は全体比較の次候補として残す。今回の実装は最初に全体検収したlane数・条件付き更新に限る。

## 生成全体と統合版

通常/QAT、英語31 token・日本語37 token、容量128、最大64生成。既存parallel/dense/融合/投入768を基準に、
各モデルでABBA×2、各ロード2暖機と本計測3回×英日、計160生成を実行した。
各入力・ロードの本計測3回中央値をまとめる。
私有実験では両方式のdeviceにsubgroup機能を要求し、候補だけWGSLを差し替えた。製品UIでは候補にだけ機能を要求する。
GPU取得・モデルロードはdecode計時の外だが、両走行の環境が完全に同一だとは扱わない。

| モデル  |  既存parallel |  subgroup候補 |   比率 | ABBAごとの比率  |
| ------- | ------------: | ------------: | -----: | --------------- |
| 通常E2B | 108.846 tok/s | 111.901 tok/s | 1.0281 | 1.0214 / 1.0241 |
| QAT E2B |  95.383 tok/s |  99.621 tok/s | 1.0444 | 1.0335 / 1.0492 |

全160生成のtoken/stop/textは方式間・反復間ですべて一致した。
統合後は製品の比較画面を初期選択の8設定80生成で実走した。

| モデル  |  既存parallel | parallel-subgroup32 |   比率 |
| ------- | ------------: | ------------------: | -----: |
| 通常E2B | 111.864 tok/s |       111.475 tok/s | 0.9965 |
| QAT E2B |  93.903 tok/s |        98.397 tok/s | 1.0479 |

通常版の利得は再現しないため、速くなったとは結論しない。QATは約4〜5%の利得が再現した。
暖機後のTTFTはほぼ不変（QAT英語37.678→37.335 ms、日本語41.483→41.445 ms）。
80生成も私有実験の既存parallelとtoken/stop/textが一致。性能計測のために生成文を変更した結果ではない。
広い品質評価を追加したわけではなく、E4B・長文・実モデルMTP・他GPUの検収は残る。

製品からの共有数値検査は180件・28,080要素でCPU許容帯、parallelとのu32一致、M1/4/8の先頭行、未使用行の保持を確認した。
Sessionでは35実行で明示指定の診断キー・M1/4/8の一致・M9と対象外形状の非適用を確認した。
初期選択・順序・実行中の設定ロック・JSON保存・不正URLの資産取得前拒否も自動操作で通過。
UI bundleは`b098c37dc4fa88c097d772a13a689db10533846e858a04a6708145736e8466e0`。

## 次の追試と残件

導入時はM2で通常/QATの各ABBA、計8設定80生成を依頼した。[追試は完了](2026-09-13-m2-gemv-subgroup-adoption.md)し、
現在の画面の初期選択は既存parallelの2設定20生成へ戻す。GEMVの比較中はRMSをworkgroupに固定する。
前回のRMS subgroup比較やTransformers.jsは選択肢として残す。
Deno 2.9.6は必要な機能を提供していないため、新経路はChromeで任意指定する。共通の既定quantにはまだ追加しない。
次の独立候補は大きい行列のlane数、SRQを含む複合処理、長い生成でのCPU/GPU待機の再帰属。

## 生データ・失敗の保存

すべて`outputs/bench/karume/`配下。再実行は新しい専用ディレクトリを使い、既存成果物を上書きしない。

| 内容                         | ディレクトリ                                            |
| ---------------------------- | ------------------------------------------------------- |
| M2の添付・集計               | `2026-09-13_m2-rms-subgroup-retry-miza863d`             |
| 現構成のCPU/GPU計測          | `2026-09-13_post-rms-profile-x4rz26uw`                  |
| プロファイルの集計           | `2026-09-13_post-rms-summary-j4re6zkg`                  |
| GEMVの最初の単体比較         | `2026-09-13_gemv-subgroup-micro-retry-56uyr3vx`         |
| 160生成の全体比較            | `2026-09-13_gemv-subgroup-whole-q3lq89i_`               |
| 追加幾何の有効な候補         | `2026-09-13_gemv-butterfly-valid-a7p0q1__`              |
| 統合UIと数値検査             | `2026-09-13_gemv-subgroup-ui-ct5iarma`                  |
| 最終共有検査と上限超過の検証 | `2026-09-13_gemv-subgroup-final-check-jj2su5_j`         |
| 集計スクリプト               | `2026-09-13_gemv-subgroup-report-6j8h5ivv/summarize.py` |

準備・失敗も保存した。M2集計の初回`2026-09-13_m2-rms-subgroup-ir2hno9a`はstop欄の名前の誤り。
最初のmicro `2026-09-13_gemv-subgroup-micro-s7hgnsmi`はハーネス終了処理が存在しない`gpu.dispose()`を呼び、結果を保存できなかった。
別ディレクトリで`destroy()`へ修正して再実行した。数値条件は変更していない。
`2026-09-13_gemv-subgroup-butterfly-5ctjz6f8`と`2026-09-13_gemv-butterfly-diagnosis-zbp9ztq8`は上述の投入数上限超過。
誤った出力0は両方式の無効commandに由来する。計測に必要な上限検査を探索コードへ追加し、無効形状をJSONへ記録した。
製品は既存の`tiledWorkgroups`の拒否を維持する。機能取得前のpage初期化は最大2回に限り履歴を保存した。
検証helperの明示throwをfinallyから外すlint修正後、共有数値検査を最終コードから再bundleして実走した。

## 全体検証

初回の全体走行`2026-09-13_gemv-subgroup-verify-rw3p528c`は、差分レビューで新規テストの期待の誤りを見つけて中断した（exit 130）。
RMSのテストから複製したnull拒否期待が残っていたが、既存の`linearGemvReduce`はnullishを省略値として扱う。
既存APIや既存テストを変更せず、新規テストだけを修正した。中断理由は同ディレクトリの`interruption.json`。
修正後の部分検証は102 passed（5 steps）、0 failed（機能未提供の実走1 stepは明示SKIP）。
ログは`2026-09-13_gemv-subgroup-focused-f3r07461/test.log`。

2走目`2026-09-13_gemv-subgroup-verify-retry-v495elah`も、最終レビューの修正を反映するためSIGINTで中断した。
中断時のDeno終了コードは139だった。ログはSIGINTと実行中の`e2e_gemma4_sequence_test.ts`を示す。
中断前のテスト失敗は記録されていないが、通常完走とは区別し、このファイルも単独実行で確認する。
追加テストの適用箇所診断が要求するtimestamp-queryを実行条件へ明記し、初期選択変更後のHTMLのバケット比較手順を補足した。
検証時の機能不足を誤って失敗とせず明示SKIPするための条件で、数値の期待値・許容誤差は変えない。

単独再検証は2 passed（7 steps）・0 failed、11秒で成功。ログは`2026-09-13_gemv-final-focused-501vn166/test.log`。
最終の`deno task verify`は**2,959 passed（802 steps）・0 failed・5 ignored（2 steps）**で完了した。
テスト部分は25分51秒、verify全体は1,553.624秒。ログ・終了コード・検証開始時のソースSHAは
`2026-09-13_gemv-subgroup-verify-final-8okk9vns/`。
Denoで追加の1 stepをSKIPしたのは必要なsubgroup機能が未提供のため。Chromeでは同じ数値検査とSession診断を実走済み。

最終HTMLの説明修正後も、サーバーの初期選択と推論bundleのSHAを再確認した。
`2026-09-13_gemv-final-browser-static-tlus1k_v/result.json`・`config.json`に保存し、UIの80生成を検収したbundleと一致した。
この再確認ではGPUを使っていない。検証後に推論コード・テスト・WGSL・画面の実行設定は変更していない。
exporter / recipeは変更していないため、pytestは今回の対象に含めない。
