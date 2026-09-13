# M2の融合追試とRMS縮約候補の比較

2026-09-13の時点スナップショット。利用者のM2結果と、RTX 3080 Tiの実験・統合検収を区別する。

## 利用者のM2結果

添付`gemma-browser-2026-09-13T11-58-16.422Z.json`を受領。
SHA256は`c2daa8d6523a0805a6738df8d850b7bbcc857a6f16376ab678e446e020bdb68f`。
保存先は`outputs/bench/karume/2026-09-13_m2-norm-adoption-zfi7023c/`。
通常/QAT E2B、parallel・dense・capacity128・chunk64、3設定往復、計12設定120生成。
同じ入力・モデルでtoken列・stop・textは全設定と反復で一致し、前回の対応する参照生成とも一致した。

| モデル  | 従来・上限1024 | 上限768のみ | 融合＋上限768 | 従来比 |
| ------- | -------------: | ----------: | ------------: | -----: |
| 通常E2B |         28.966 |      29.301 |        30.043 | +3.72% |
| QAT E2B |         31.246 |      31.901 |        32.046 | +2.56% |

単位tok/s。各設定の2ロード×2入力について本計測3回の中央値を取り、その4値の中央値。
融合だけの追加利得は上限768に対して通常約2.53%、QAT約0.45%。
前回の約32.6/34.5 tok/sより絶対値が低いが、温度・電力・クロックの記録がなく原因未確定。
**前回と今回のbundleは異なる**。今回のbundleは`e8679798de0615130971b3782f66e254c2907864e4bdc49c9f6141c71f6a7139`、checkoutは`7114d0c`。
参照RMSのキー・WGSLは維持されているが、それだけで環境差と断定しない。

K-36のM2追試は完了。融合＋上限768は次候補の共通基準にする。
モデル既定への組み込みは別段階。投入上限は既存ADR 0038のホスト政策であり、現在のquant保存語彙にsubmitPolicyはない。
この値を無説明で配布者の設定へ移さず、融合のquant選択とホストの投入方針を分けて判断する。

## RMS単体の候補

3幅（256/1536/2560）×5行数（1/8/32/40/64）を実重みの係数で測定。
各候補はsin・広い値域・特殊値の3パターンと通常/強制grid-stride、90検査・3,786,240要素。
計時前にGPUを暖機し、6回の測定順を反転する。単体台本のmedianは6値の上側中央値であり、中央2値の平均ではない。

| 候補                                   | 数値検査                         | 性能と判断                                                              |
| -------------------------------------- | -------------------------------- | ----------------------------------------------------------------------- |
| 元の256部分和を128/64 threadへ分配     | 有限u32・NaN分類の差0            | 128は中央値0.646倍（0.445〜1.006）、64は0.408倍（0.316〜0.798）。見送り |
| vec2/vec4 load/storeで元の加算木を保持 | 差0                              | vec2の利得は幅依存、vec4は悪化する形もある。一括採用しない              |
| subgroupShuffleXorで元の木を保持       | 差0                              | 幅1536で単体約1.04倍、モデル全体は約1%。追加採用を見送る                |
| subgroupAdd・scalar、32レーン固定      | 有限u32の差392,032、NaN分類の差0 | 幅256で約1.24〜1.30倍、1536で約1.14倍、2560で約1.10倍。任意経路へ       |
| subgroupAdd・vec2/vec4                 | 下位bitが変化、NaN分類の差0      | scalarを上回る一貫した効果なし                                          |
| 1 subgroup/行・vec4                    | 下位bitが変化、NaN分類の差0      | 幅1536/2560で遅い。見送り                                               |

scalar subgroupAddの最大絶対差は`1.52587890625e-5`、最大相対差は`2.3761466244045228e-7`。
全GPUでの誤差上限ではなく、この固定コーパスの観測値。
Deno 2.9.6はsubgroupsを提供しないため、後半の候補はChrome 153で測った。
M2添付にはsubgroupsとsubgroup-size-controlの広告があるが、新経路の実走証明にはならない。

## モデル全体と帰属

全設定でparallel・dense・融合あり・上限768を揃え、RMS単体と融合RMSの両方を切り替えた。
私有ソースコピーで参照→加算木保持→subgroupAdd→subgroupAdd→加算木保持→参照。
通常/QAT E2Bの12ロード・120生成。全設定で同じGPU機能を要求した。
通常の最大値選択・転送・生成器を使い、本文decodeは速度計時外。

| モデル  |    参照 | 加算木保持 | subgroupAdd | subgroupAddの利得 |
| ------- | ------: | ---------: | ----------: | ----------------: |
| 通常E2B | 111.742 |    112.712 |     111.821 |            +0.07% |
| QAT E2B |  92.600 |     93.481 |      96.088 |            +3.77% |

通常は全生成が一致。QATはsubgroupAddの20生成で文章が変わり、同設定の反復内では一定。
確認した英語・日本語の短い助言リストは意味のある文章だったが、この観察を広い品質保証にはしない。

続いて、参照が選んだ同じ64トークンを次入力へ与え、PLE行・本文decode・長さの違いを揃えた。
GPUの計算・最大値選択・転送は実行した上で、私有生成器の選択結果を固定tokenへ置き換える。
モデルごとに参照→候補→候補→参照、計8ロード80生成。

| モデル  |    参照 | subgroupAdd |   利得 |
| ------- | ------: | ----------: | -----: |
| 通常E2B | 111.803 |     112.040 | +0.21% |
| QAT E2B |  93.245 |      98.993 | +6.16% |

この値は**入力列を固定した帰属実験**で、通常生成の速度ではない。
QATの利得は異なる生成token列だけでは説明できない。ただし内部値・SRQの値依存やCPU/GPUの重なりまで固定する実験ではない。
RMSの単体利得から、全体の各費用を厳密に逆算しない。

## QATの小規模品質

既存の固定ARC 64問とWiki 29窓・1,870予測tokenを、参照/候補の同じ入力列で評価した。
capacity256・chunk64、全語彙logitsを読み戻し、JavaScriptのfloat64でlog-sum-expを計算。
速度の測定ではない。既存コーパス・選択肢・期待答を変更していない。

| 指標                       |     参照 | subgroupAdd |
| -------------------------- | -------: | ----------: |
| ARC正答数                  |  44 / 64 |     45 / 64 |
| 選択肢長で正規化した正答数 |  43 / 64 |     44 / 64 |
| Wiki平均負の対数尤度       | 6.500792 |    6.493218 |
| Wiki perplexity            |  665.669 |     660.646 |

予測した選択肢が変わった問題は3問。今回の差に大幅劣化は認めないが、改善の統計的証明には小さすぎる。
この短い窓のperplexityの絶対値は高く、長文・標準設定の品質スコアと同一視しない。
他GPU・E4B・投機生成全体への外挿はしない。CPU/PyTorchとの新たな一致を主張しない。

## 統合と次のM2検収

[ADR 0100](../decisions/0100-rms-subgroup-reduction.md)に従い、runtimeとGemmaに`rmsNormReduce: "subgroup32"`を追加。
GPU取得は`acquireGpu({ subgroups: true })`。機能不足は拒否し、元のキー・WGSL・goldenを保持する。
モデル既定・quant・CLI・配布形式は維持する。Denoでは明示指定しても利用できないため、Chromeで検収する。

統合版のChromeは61ケース・688,307要素のCPU参照/先頭行/融合/寿命/端数幅/強制grid-stride検査を通過。
UI初期選択の40生成も、それぞれ対応する私有実験の経路と一致した（両方式どうしの一致ではない）。
QATのABBAは99.594→101.645 tok/s、+2.06%。参照の4中央値は95.041〜100.924、候補は101.180〜101.797。
単一の改善率として固定せず、通常生成では約2〜4%の上積みが観測されたと扱う。
UIの初期選択・実行順・設定ロック・JSON保存・不正URLの資産取得前拒否も自動操作で検収した。
暖機後のTTFTは2ロードの中央値をさらに中央2値の平均でまとめると、英語36.703→36.515 ms、日本語41.188→40.008 ms。
初回英語のTTFTは参照298.610/411.860 ms、候補486.285/420.345 ms。ロード時間も初回だけ短く、
初回値の幅と実行順の影響を分離できていないため、冷間の改善は主張しない。

次はM2で`deno task bench:llm-browser`を起動し、初期選択のまま「計測開始」。QATの4設定40生成だけでよい。
通常版の追加計測も選択できるが、RTXでは全体利得がなかったため初期選択から外した。
その結果でChromeでの採用範囲を判断し、次の独立候補は更新後のプロファイルに基づいて選ぶ。

## 生データと再現

すべて`outputs/bench/karume/`配下。再実行も新規ディレクトリとし、既存の準備コード・結果は変更していない。
集計は[保存JSON](2026-09-13-rms-subgroup-reduction-results.json)と`2026-09-13_rms-report-pwwrt85b/summarize.py`。

| 内容                    | ディレクトリ                                  |
| ----------------------- | --------------------------------------------- |
| M2添付・検算            | `2026-09-13_m2-norm-adoption-zfi7023c`        |
| thread配置              | `2026-09-13_rms-geometry-micro-s09nizj5`      |
| vector                  | `2026-09-13_rms-vector-micro-tfexh8z3`        |
| Deno機能調査            | `2026-09-13_rms-subgroup-capability-pjnpn3g2` |
| Chromeの元の加算木      | `2026-09-13_rms-subgroup-browser-vp1vkjzz`    |
| ChromeのsubgroupAdd候補 | `2026-09-13_rms-subgroup-fast-mgaiywfu`       |
| 全体120生成             | `2026-09-13_rms-whole-browser-uw_k66jo`       |
| 固定入力列80生成        | `2026-09-13_rms-replay-browser-zxjf_ith`      |
| 上記2件の集計           | `2026-09-13_rms-summary-k93hog_5`             |
| QAT品質                 | `2026-09-13_rms-subgroup-quality-9lvtm0cg`    |
| 品質集計                | `2026-09-13_rms-quality-summary-51ifwwhz`     |
| 統合UI・数値検査        | `2026-09-13_rms-subgroup-ui-lf0xylkp`         |
| Deno部分検証            | `2026-09-13_rms-focused-retry-6kw7jo9w`       |

準備失敗も保存した。`2026-09-13_rms-subgroup-micro-w_ph1n6f`はDeno型のwgslLanguageFeatures不足で実走前に終了。
`2026-09-13_rms-focused-tests-0jgsf2ak`は編集時のimport位置の誤りで型検査が終了し、別出力先の再走で修正を検証した。
Chrome起動時のInstance参照消失は、生成前の最大2回のpage初期化として履歴を保存。計測結果を選別する再試行はしていない。

## 全体検証の切り分け

初回の`deno task verify`は2,957 passed（797 steps）・1 failed・5 ignored（1 step）、25分48秒。
失敗は`e2e_gemma4_pretrained_test.ts`の実DL検査で、GPU取得より前のadmissionが「グラフ出力1本」を拒否した。
同じファイルの単独実行は3 passed（5 steps）、0 failed、10秒。コード・期待値は変更せず、キャッシュの削除・修復はしていない。
現物のグラフはmanifest記載のSHAと一致し、logits・hiddenの2出力を持つ。
GPUメモリ不足や新しい縮約の誤差を原因とする証拠はない。

永続キャッシュを読み取り専用で調べると、テスト用`karume-local/dist`の古いmanifestと1出力グラフが残っていた。
`serveLocalDist`は固定repo・全0のrevisionを使い、manifestのキャッシュキーはポートを含むresolve URL。
同じポートが再利用される条件をメモリ内のキャッシュと模擬fetchで再現すると、配信内容を更新しても旧manifestを読み、
別ポートでは新manifestを取得した。GPUや実キャッシュの更新は使っていない。
**当該失敗のポートは未記録なので、今回の原因への帰属は推測**。テスト用サーバーの世代識別の別件として
[known issues](../known-issues.md#疑似hfサーバーの固定revisionで古いmanifestが再利用される)へ残す。
最適化の実装へキャッシュ削除・自動再試行・admission緩和を混ぜない。

- 初回全体: `2026-09-13_rms-subgroup-verify-a4sat_jg/verify.log`
- 単独実行: `2026-09-13_rms-pretrained-isolation-23leknsz/test.log`
- 読み取り専用の索引: `2026-09-13_rms-cache-audit-qqq__7cf/metadata.json`
- 旧manifestとグラフ: `2026-09-13_rms-cache-manifests-yf0rcaep/manifests.json`、`2026-09-13_rms-cache-graphs-2muawztn/graphs.json`
- 永続キャッシュを書き換えない再現: `2026-09-13_rms-local-revision-repro-1xr6o1wz/run.py`・`result.json`

再走の`deno task verify`は**2,958 passed（800 steps）・0 failed・5 ignored（1 step）、25分50秒**で完了。
ログは`2026-09-13_rms-subgroup-verify-retry-_ylusnzs/verify.log`、終了コードと計測時間は同ディレクトリの`completion.json`。
Denoの追加SKIPは機能未提供のsubgroup実走1項目。Chromeの同じ検査は61ケース・688,307要素を実走済み。
全体再走でも失敗した配布形読み込みテストは成功した。確認後に推論コード・テスト・WGSLは変更していない。
既存RMSコード・既存fixtureの変更0、文書リンク423件の欠落0を確認した。
exporter / recipeは変更していないため、今回の検証にpytestは含めない。
