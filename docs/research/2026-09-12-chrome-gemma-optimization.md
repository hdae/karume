> 2026-09-12時点のChrome実測と最適化実験のスナップショット。RTX 3080 Tiの結果であり、Apple M2の実測ではない。

# ChromeでのGemma速度差の帰属とRoPE融合

[WebMLとの同条件比較](2026-09-12-webml-browser-speed.md)に続き、既定の温度0・非投機`pipeline.chat`をプロファイルした。
CPU側だけでは差を説明できず、GPU内の行列計算と多数の小カーネルが主な改善対象だった。
重みの読み方・実行形状・待ちの重複・RoPE融合を分離して試し、まずRoPE融合を採用する。
[集計・各実験の出典hash・生成列](2026-09-12-chrome-gemma-optimization-results.json)を併記する。

## 条件と外部ソース

基準の製品コードは`a0d6d13`（調査開始HEADは文書コミット`a295ede`）。Deno 2.9.6、Chrome 153.0.8010.36、
Linux / RTX 3080 Ti / Vulkan、Xvfb併用headless。容量8,192、同一41入力token、chunkLength=64、
f32計算、通常版とQAT E2Bのローカル配布物、既定PLE予算、文字列復号込み。
モデルロードを除き、TTFTは最初のtoken通知まで、decodeは残りtoken数をその後の終了までの時間で割る。

指定された[WebMLの公開JavaScript](https://webml-community-gemma-4-webgpu-kernels.static.hf.space/gemma-4-e2b.js)を再取得して読み、
既存の参考用チェックアウトとSHA-256が一致した（551,802 bytes、詳細は実験の`webml-source.json`）。
WebMLは専用実装で、複数tokenのGPU先行実行、融合、異なる行列計算経路を持つ。
第三者の実装ソースは製品へ複製していない。今回の候補はKarumeの既存コードから作成した。

## 既定経路のプロファイル

Chrome DevTools ProtocolのCPUサンプリングでは、256token走行の大半がidleだった。
QATは4,240 ms中3,919 ms、通常版は4,825 ms中4,504 ms。
JSの実行計画・エンコード・文字列復号だけを削っても、数倍の差を埋める説明にはならない。
WebGPU APIの観測では、定常1tokenあたりsubmit=4、compute pass=3、writeBuffer=8、mapAsync=1、error scope終了=2。

外部GPU timestampを既存passに置き、prefillを除いた63decodeを集計した。

| E2B  | GPU ms/token | 全体 ms/token | dispatch/token |
| ---- | -----------: | ------------: | -------------: |
| QAT  |       12.322 |        15.986 |          1,862 |
| 通常 |       13.485 |        17.151 |          1,375 |

この走行ではGPU時間が全体の約77〜79％。待ち時間にはGPUの処理待ちも含まれ、
`popErrorScope`の所要時間をそのまま「検査の余分な費用」とは扱わない。並行する2検査は区間の和集合で数える。

dispatch別の順位は、別走行で1dispatchごとにpassを区切って求めた。
QATはSRQ（固定スケールへの再量子化）487回、INT2行列積61回、INT4 g512行列積95回、RMS正規化242回が上位。
通常版はINT4 g32行列積276回、RMS正規化242回、strided処理205回、INT8語彙出力1回が上位。
**pass分割でGPU時間自体がQAT 20.65 / 通常19.73 msへ増える**ため、この順位を通常走行の厳密な割合へ換算しない。

最初のtimestamp実験は、feature追加によってruntimeの診断経路も有効になり、既定greedyから外れた。
さらにtimestampの二重開始で失敗したため不採用として保存した。
修正版では実験用probeだけでruntime診断を無効化し、外部timestampと分離した。製品のエラー検査・読み戻し処理は保持した。
診断の有無を含む各生成列も基準と一致した。

## 個別候補の採否

### 検査と読み戻しを重ねる案: 不採用

`finishAndRead`でerror scopeの終了とmapAsyncを同時に開始し、両方の成功後だけ値を読む候補を隔離して比較した。
エラーの伝播を残し、A/B/B/A順・各256token×3回・通常/QATで計24走行は出力一致。
しかし候補の改善は安定せず、通常版で逆効果の組もあった。製品の同期処理は変更しない。

### GEMVの重み配置: 有望、未統合

GEMVは1行の入力と重み行列の積。既存のK方向の加算順を保ち、重みの32 bit語だけを並べ替える候補を比較した。
INT2 / INT4 g32 / INT8、6形状、4配置、3種類の入力・biasで216組のu32一致を確認した。
計時は1pass内64dispatch、暖機用GPU処理を各試料の前に置き、8回の順序交替で比を求めた。

- 通常版のFFN拡張形（INT4、N=6,144 / K=1,536）は約1.70〜1.74倍。
- INT8の語彙出力形（N=262,144 / K=1,536）は約2.49〜2.64倍。
- INT2は多くの形で数％程度。並べ方によっては遅くなる。

これは合成入力のカーネル単体値。既存資産はrow-majorで、prefillのGEMMや埋め込みも同じ重みを使う。
追加GPUコピーのメモリ・寿命・共有・見積りを設計する必要がある。語彙INT8だけでも追加コピーは約384 MiB。
全体速度やM2の効果は未検収。保存形式を黙って変更しない。
実IRでは語彙INT8は埋め込みとも共有される一方、FFNのINT4はlinear専用。6,144×1,536の30本を二重化すると追加payloadは135 MiB。
より多い12,288×1,536の40本（360 MiB）は最初の単体比較に含めていないため、次に検証する。
`outputs/bench/karume/2026-09-12_gemv-layout-memory-plan-3pv7pdyr`に消費先・所有・共有・見積りの境界を記録した。

### GEMVのworkgroupサイズと展開数: 有望な限定形あり、未統合

同じ18形状でスレッド数4〜128とループ展開1〜8の候補を比較し、639組のu32一致を確認した。
大語彙INT8は32→16スレッドで約1.45倍。小さいINT8の一部は展開4→8で改善したが、
INT2や長いINT4には低下する形もあり、既定を一律には変えない。
語彙出力だけの形状選択は追加メモリを要さず、次の全体比較候補とする。

### BSHD RoPE融合: 採用

RoPEは位置情報を加える回転演算。旧matcherは`[1,H,S,D]`と表`[1,1,S,D]`を受理したが、
Gemma 4は`[1,S,H,D]`と表`[1,S,1,D]`を発行する。
このためdecodeではH=1の所有Kの15鎖だけ、prefillでは0鎖しか融合されていなかった。
演算の順序は既に対応しており、**表を引く軸の不一致**が原因だった。

位置表の添字をBHSDでは`row % S`、BSHDでは`row / H`として別キーで生成する。
7ノード（slice×2、neg、cat、mul×2、add）の8dispatchを1へ畳む。
既存の積の丸め障壁、半分割、入力の結線、privateな中間値、broadcast形状の検査は維持する。
既存BHSDのWGSLはbyte不変、新BSHDも計測候補とbyte一致を確認した。
公開API・IR・重み・計算精度・サンプリングは変更しない（[ADR 0040](../decisions/0040-fusion-pass.md)）。

Gemma通常/QAT E2Bはdecode / prefillとも50鎖を融合する。
統合後のChrome実測でも、decodeのdispatchはQAT 1,862→1,617、通常1,375→1,130（各−245）となった。
pass数は3のまま。追試は暖機条件が異なるため、その絶対時間を前のプロファイルとの差分評価には使わない。
同じ軸順のAnima text conditionerも0→24鎖へ増える。実グラフは16heads / D=64、Tsrc/Ttgt別の鎖であり、
この形もGPUテストへ追加した。既存の融合カウンタ検査の期待値は、この受理集合の拡大を反映したもの。
Animaの画像生成全体の速度向上は今回単独では測っていない。

最初は各armごとにページを作り直し、A/B/B/A×256token×3回を実行した（計24走行）。
暖機後でも時刻による変動が大きく、通常版の集計差は小さかった。
次に同じChrome内の2ページにA/Bを保持し、GPU実行は必ず1ページずつ、各3回の暖機後にABBAを4組実行した。
同じプロンプトを繰り返すためPLE行キャッシュが温まった条件であり、初回ターンとは区別する。

| E2B  | ABBA各組の改善倍率                | 倍率中央値 |
| ---- | --------------------------------- | ---------: |
| QAT  | 1.0307 / 1.0276 / 1.0312 / 1.0280 |     1.0293 |
| 通常 | 1.0276 / 1.0323 / 1.0259 / 1.0138 |     1.0268 |

QATは約62.5〜62.9→64.3〜64.8 tok/s、通常版は約55.3〜56.3→56.4〜57.8 tok/s（各ABBA組のA/B平均）。
暖機後TTFTの中央値はQAT 68.1→62.75 ms、通常76.4→72.7 ms（各変種8走行）。
**計56走行でモデル内の全token列・文章・停止理由が一致**した。
さらに、先行して実行した英語・日本語の計16走行もA/Bで一致した（QAT 281 / 745 token、通常389 / 794 token）。
2ページ常駐によるメモリ条件の違いがあるため、最初の単ページ結果も保存した。
この小幅改善をWebMLとの差の解消とは扱わない。M2の効果と数値一致は実機追試が必要。

## 次の融合で維持すべき条件

参考実装にはK方向を複数laneへ分けた部分和の合算と、正規化→SRQ→gate/up、残差加算→次の正規化→SRQなどの融合がある。
前者はKarumeの既定の逐次K加算とは数値契約が異なるため、将来の明示opt-inと品質比較を条件に検討する。
後者は小dispatchと中間バッファを減らす候補になる。

ただし一部の尾部融合は、全workgroupの完了counterを使って最後のworkgroupに後段を実行させる。
WGSLのatomicは異なるメモリ位置間の順序を保証せず、barrierの同期範囲もWorkgroupである。
この組合せだけで他workgroupのpartial配列の可視性を可搬に証明できるとは扱わない。
[WGSL 2026-08-31版 §14.5.3–4 / §17.8](https://www.w3.org/TR/2026/CRD-WGSL-20260831/#memory-model)を根拠とする推論であり、
WebMLで誤出力を実測したという意味ではない。Karumeの候補では別dispatchで依存を作るか、同じworkgroup内に処理を閉じる。
出典と読み取り位置は`outputs/bench/karume/2026-09-12_webml-fusion-contract-survey-g1hkfudx`に記録した。

### GPU常駐PLEと複数tokenの先行実行

参考実装は生成tokenをGPU上の次回入力へ繋ぐ経路を持つ。KarumeではPLE（層ごとの補助埋め込み）をホストで準備しているため、
単にsubmitをまとめるだけでは同じ構成にならない。現行資産のPLE総量は通常E2B約2.22 GiB、QAT E2B約1.13 GiB。
GPUへ置けばCPUを経由する箇所を減らせる可能性があるが、追加メモリ・binding上限・丸め・runリース・EOS/中断時の状態を設計する必要がある。
まず1tokenのGPU内PLE入力、次に複数tokenの先行へ分ける案を、`outputs/bench/karume/2026-09-12_gemma-gpu-ple-survey-aszhtwxh`へ記録した。
現行のメモリ予算と生成動作には変更を入れていない。まだ実測していない案であり、GPU計算そのものの短縮とは分けて評価する。

## 検証と出典

統合後の重点検証は113 passed / 0 failed。追加した通常/QAT配布グラフを含む資産検査も9 passed / 0 failed。
RoPEのGPU検査は両発行順、head/sequenceの変化、D=6/64/128/256/512、端数、±0、非正規化数、Inf/NaN、
workgroup上限2でのgrid-strideを含む。有限値はu32一致、NaNは既存の分類一致基準を保つ。
従来のBHSD snapshotを更新せず、新BSHD snapshotを追加した。

実験はすべて`outputs/bench/karume/`下の独立ディレクトリへ保存した。

- CPU/API: `2026-09-12_chrome-native-profile-tgfwws7w`
- GPU既定経路: `2026-09-12_chrome-gpu-native-profile-h20aerev`（失敗した初案は`chrome-gpu-profile-fkj8su05`）
- 同期待ち: `2026-09-12_chrome-map-overlap-fxx5x8kh`
- 重み配置: `2026-09-12_chrome-gemv-layout-zj5wocyn`
- GEMV形状: `2026-09-12_chrome-gemv-geometry-_7r4hgi5`
- RoPE隔離候補・単ページ比較・重点検証: `2026-09-12_chrome-gemma-rope-fusion-p491ln7m`
- RoPE短間隔の交互比較: `2026-09-12_chrome-rope-interleaved-8qj49eum`
- 統合後のdispatch再集計: `2026-09-12_chrome-rope-gpu-census-tz3rh8f9`

各候補と実測のhash、採用しなかった実験、集計に使った各走行の速度は保存JSONから辿れる。
追加実験は既存の結果を上書きせず、新しい出力先で行う。

全体検証の初回は2,941 passed（771 steps）/ 3 failed / 5 ignored、25分9秒。
3件は`fusion-hints` / `opbench census`が旧RoPEヒット15・未対応35を固定していたためで、単独実行でも再現した。
実グラフとGPUのdispatch再集計を根拠に、対応50・未対応0へ更新した。数値許容差・goldenは変更していない。
修正後の両ファイル単独実行は19 passed / 0 failed。ログはRoPE実験ディレクトリの`tools-isolated-{failure,fixed}.log`。
初回全体ログは`outputs/bench/karume/2026-09-12_rope-fusion-verify-8n7rnbso/verify.log`に保持した。

修正後の全体検証は**2,944 passed（771 steps）/ 0 failed / 5 ignored、25分4秒**。
`outputs/bench/karume/2026-09-12_rope-fusion-verify-census-jw476oh1/verify.log`と`completion.json`に保存した。
Gemmaの投機検証先頭行とdecodeのu32一致、既存の画像・音声golden、追加したBSHDの門も通過した。
