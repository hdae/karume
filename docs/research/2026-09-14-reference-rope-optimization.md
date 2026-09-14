この文書は2026-09-14時点の添付資料の再検証とRTX 3080 Ti実測のスナップショットである。

# Gemma E2Bの参照資料再検証とRoPE周辺のコピー削減

添付資料を現行`c265a60`と照合し、RMS正規化→RoPEの融合と、要素順を変えない`permute`のコピー削減を比較した。
前者は追加カーネルに対して全体利得が小さく保留。後者は既存の別名管理だけでdecodeの100 dispatchを省けるため採用する。
所有権保護を含む最終実装のChrome反復では通常E2Bが約2.44%、QAT E2Bが約0.93%速かった。ただし別走行のQATでは差が逆転し、一定の改善率を保証できない。

## 資料と比較基点

利用者の`gemma4_e2b_reference_bundle.zip`のSHA-256は
`b6637fb8b9a8431e02234a869342dcc71d688f2b5cb3ec44678085267b65fbac`。
CRCと全memberのSHAを確認し、同梱のKarume参照20ファイルを現行と比較すると14変更・6同一だった。
資料自身の検証は抽出・静的確認までで、カーネルのGPU実測ではない。

同梱の元`gemma-4-e2b.js`は551,802 bytes、
SHA-256は`0234c0e866bfaa9623e938a7cfa7f5740cca22532cc1112dd4e8915b97f78d62`。
同梱MIT LICENSEを外部カーネル全体のライセンス根拠とは扱わず、外部コードや付属検証プログラムを実行・製品へ複製していない。
候補は現行Karumeの数式・カーネルを基に独立実装し、実験ディレクトリへ隔離した。

監査の正本は`outputs/bench/karume/2026-09-14_22-19-05_reference-bundle-audit-fi7jkgqu/`の
`archive-audit.json`、`census.json`、`PLAN.md`。
現行では既にparallel GEMV、固定QAT/INT2/SRQ、RMS→add、BSHD RoPE、GPU greedyの8B読戻しがある。
古い比較表の「未対応」をそのまま現行の不足とは扱わない。
転置した重みの複製も[K-33の費用比較](2026-09-12-chrome-gemv-parallel.md#その他の採否)を確認する。今回その複製は採用しない。

## 条件と集計方法

数値の機械可読な正本は[結果JSON](2026-09-14-reference-rope-optimization-results.json)。
集計プログラムと出力は`outputs/bench/karume/2026-09-14_23-20-48_reference-rope-owned-summary-4p8iwz85/`。

- RTX 3080 Ti、Chrome 153.0.8010.36、Vulkan、Xvfb。単体はDeno 2.9.6。
- 通常Gemma4 E2Bと固定mobile QAT E2B。新しいローカル配布の`i4-gemvpar`、
  f32計算、parallel GEMV、RMS→add融合、投入上限768、capacity128、dense chunk64。
- 英語・日本語の2入力、温度0・GPU greedy、各64 token。
  各ロードで入力ごとに2回暖機＋3回測定。TTFT（最初のtokenまでの時間）とdecodeを分ける。
- decode tok/sは`63,000 / (elapsedMs - ttftMs)`。
  まず各ロード・入力の3測定の中央値、次に同じ実装・モデルの中央値を取る。
  全生成の中央値へ暖機や初回コンパイルを混ぜない。
- 統合初版と所有権保護後の最終実装は各モデル8ロードで基準/候補をABBA+BAAB順に交互実行し、計160生成。
  基準は変更前の凍結bundle、候補はimport-map差し替えを使わない実製品bundle。
  SHA・manifest・実行設定・各測定をJSONに記録した。feature集合は列挙順だけを正規化して照合した。
- 各製品bundle作成時からその測定終了までTS/WGSL/HTML 785ファイルのSHAは不変。
  この最終生成の測定中にはテスト・他のGPU実験を並走していない。

基準bundleのSHAは`4c73e9a7de2978f5508f965467fe7ce37755501e64ff77107be1dbdd4355b1cc`。
これは現行変更前のKarumeであり、添付資料内の古いKarumeを速度基準にしていない。
添付WebML本体の全生成は今回は再測定していない。[過去の同条件比較](2026-09-12-webml-browser-speed.md)と区別する。

## K-40: RMS正規化→RoPE融合は保留

両E2Bの現行IRには50本の対象鎖があった。内訳はD256で28本の8 headと12本の1 head、
D512で7本の8 headと3本の1 head。48本が隣接し、2本はcos/sinのreshapeを挟む。
既存RMS→add 106本を維持したまま、入力M=1/4/8/32/40/64の全12プランで50本の候補適用を確認した。

試作はRMSの既存縮約順とRoPEの丸め障壁を保持し、正規化後の値をworkgroup内で共有する。
BHSD/BSHD、D256/512、6つのhead/sequence形状、通常/強制grid-stride、5入力パターンを比較した。
24形状・240比較・10,045,440要素で有限値の語とNaN分類の不一致は0。
単体の中央値は1.177〜1.411倍だった。NaN payloadの完全一致をこの融合実験の保証とは呼ばない。

| モデル  | 基準 tok/s | 融合 tok/s |    変化 | 基準/融合 TTFT ms |
| ------- | ---------: | ---------: | ------: | ----------------: |
| 通常E2B |    112.101 |    113.086 | +0.879% |     37.45 / 37.80 |
| QAT E2B |     93.011 |     93.451 | +0.474% |     39.60 / 39.55 |

80生成はtoken列・停止token・本文一致。全体差は小さく、追加の融合判定・カーネル・数値経路を維持する費用に見合うと判断しなかった。
M2等で当該領域の費用が大きいと確認できた場合を再検討条件とする。製品の公開フラグや新カーネルは追加しない。

単体の正本は`outputs/bench/karume/2026-09-14_22-24-17_rms-rope-micro-retry-onbxj8hk/`、
全体比較は`outputs/bench/karume/2026-09-14_22-27-38_rms-rope-whole-validated-iksmh8_y/`。
初回の融合判定はRMS出力の3つの内部consumerを誤って拒否し、静的適用件数0で検出・訂正した。
この未適用版を速度結果に使っていない。

## K-41: 要素順を変えないpermuteのコピーを省く

長さ1の軸を移動しても、他の軸の順番が保たれていれば、平坦なバッファ上の要素順は変わらない。
例えば`[1, 8, 1, 256] → [1, 1, 8, 256]`はコピー不要である。
現在のRoPEとattentionの周辺にはこの形が残っており、decodeで100本、M>1の測定形で30本が該当した。

`planAliases`で宣言順に別名と実体の由来を導出し、実行・メモリ見積りで共有する。
recipe builderも別名の場合はstrided copyを発行しない。
新しいstride表現や非連続Tensorは作らず、reshapeと同じretain/release・pin・resident借用へ流す。
0要素は従来経路。非単位軸の交換は、軸長が等しくても実体化する。
ただし入力・重みやその別名から最初に到達するpermuteはコピーを維持する。
統合初版の自己レビューで、入力まで別名を伸ばすと`copyOutputs`の書戻しが自己コピーとして拒否される境界を発見した。
従来の受理条件を維持するため、内部で確保した実体だけを新しい別名化の対象にした。
既存のreshape/expandの別名化と自己コピー拒否は変更しない。
実GPUの対照でも、旧別名規則は書戻しを受理し、保護なし候補は自己コピーで拒否、最終実装は受理して値が一致した。
この3対照は`outputs/bench/karume/2026-09-14_23-19-51_permute-copyoutputs-controls-tzwr7eb1/`に保存した。
両E2BのM=1/4/8/32/40/64、12プランのJSON全体は、この保護の前後で同一だった。
証跡は`outputs/bench/karume/2026-09-14_23-16-03_permute-owned-plan-equivalence-wsuqhwo_/`。
IR・dtype対応・公開設定・数値演算・量子化の既定は変更しない。
[ADR 0011追記](../decisions/0011-layout-strategy.md#要素順を保つpermute2026-09-14)を正本とする。

| 実験                              | 通常 基準→候補 tok/s |    変化 | QAT 基準→候補 tok/s |    変化 |
| --------------------------------- | -------------------: | ------: | ------------------: | ------: |
| 隔離候補 ABBA・80生成             |    111.941 → 114.338 | +2.141% |     92.832 → 94.780 | +2.099% |
| 隔離候補 BAAB・80生成             |    110.575 → 111.592 | +0.920% |     97.486 → 96.180 | −1.339% |
| 統合初版 ABBA+BAAB・160生成       |    112.350 → 114.691 | +2.084% |     92.798 → 94.135 | +1.441% |
| 最終実装・所有権保護付き・160生成 |    112.360 → 115.100 | +2.439% |     93.175 → 94.044 | +0.933% |

最終実装のTTFTは通常37.50→37.60ms、QAT39.70→39.60msで、prefillの速度向上は明確でない。
QATの逆転した走行を除外せずに残す。これらは独立な端末標本や統計的有意差の証明ではない。
今回の採用理由は、処理削減が構造的に確かで、追加コードが小さく、
同じ数値・既存の所有権管理を維持し、通常版の各反復と最終実装のQAT全体比較が改善したこと。
M2の改善率や、他モデル全体の速度は未測定である。

生データは次の各専用ディレクトリにある。

- `outputs/bench/karume/2026-09-14_22-32-50_singleton-permute-whole-035paoby/`
- `outputs/bench/karume/2026-09-14_22-37-59_singleton-permute-baab-etjftlpn/`
- `outputs/bench/karume/2026-09-14_22-46-48_singleton-permute-product-7307b_o4/`
- `outputs/bench/karume/2026-09-14_23-15-40_singleton-permute-owned-product-lqxrj4eq/`

### GPUで何が減ったか

診断付き28生成を別に実行した。通常のcompute passを維持する測定では、63 decodeの各tokenのGPU時間を合計し中央値を取った。
GPU時間にはCPU待ち・readbackを含まない。各モデル1組の補助測定であり、上の反復生成の代わりには使わない。

| モデル         | decode dispatch/token | prefill dispatch | decode GPU ms/token |
| -------------- | --------------------: | ---------------: | ------------------: |
| 通常 基準→候補 |           1,024 → 924 |      1,022 → 992 |       6.411 → 6.281 |
| QAT 基準→候補  |         1,511 → 1,411 |    1,509 → 1,479 |       7.637 → 7.459 |

decodeのdispatch数にはGPU greedyの2 dispatchを含む。
全体では両モデルとも64 tokenあたり6,330 dispatchを削減した。
別のdispatch分割診断ではstrided copyが135→35本/tokenになった。
分割診断は各dispatchにpass境界を挿入し時間を増やすので、そのstrided時間差約0.60msを通常実行の削減時間とは扱わない。
同じ診断では通常版のparallel GEMVが最大で、QATにはstatic quantizeの費用も残る。絶対時間は[結果JSON](2026-09-14-reference-rope-optimization-results.json)で分離している。

正本は`outputs/bench/karume/2026-09-14_22-35-47_singleton-permute-profile-x1khfwbm/`。
GPU queryのresolve/mapは生成の時間計測終了後に実施した。

## 正確性・失敗の切り分け・検証

比較実験560生成と診断28生成、計588生成で、各モデル・入力のtoken列・停止token・本文が同じ。
通常版とQATのモデル間一致を意味しない。短い2入力の結果を広い品質評価と呼ばない。

追加テストは以下の故障を検出する。

- rank 1〜4、軸長1/2/3の全2,127組を、出力座標から入力の添字を独立計算するoracleと照合。
  M=1/4/8/1/32の束縛変更、0要素を非適用とする境界も検証。
- GPUのf32出力shape・語を、符号付き0・非正規化数・±Inf・NaN payloadを含めて比較。
  先頭の全区間sliceで語を保って内部値を作り、permute自体は共有時0 dispatch、実際の転置時1 dispatchを確認。
- 一時値の後続consumer・別名chain・複数出力・再実行と、resident initializerのコピー維持・寿命を検証。
- メモリ見積りは共有時24B、非共有時48B。既存の融合件数は変わらない。
- i32/boolのpermuteは現runtimeで非対応のまま。別名化できる形でもSession構築で拒否する。

- 入力→reshape→permute→reshapeの出力を同じ常駐入力へ書き戻し、2回の実行で受理と値を検証。

所有権保護後は既存resident batchテストも含め、**125 passed（9 steps）、0 failed、7秒**。
`outputs/bench/karume/2026-09-14_23-12-45_permute-ownership-tests-bd0vtg9r/`にログと12プランの適用件数を保存した。
追加で内部の0要素を非適用とするCPUテスト、融合候補列挙の既存8＋10テストも成功。
融合候補ツールの未融合プランも同じ`planAliases`を使うよう更新した。既存の期待値は変更していない。

最初の追加GPUテストはi32/boolが実行可能と誤認して2 step失敗した。
既存のcapability契約を確認し、新しいテストを拒否検証へ訂正した。既存の許容誤差やテストは弱めていない。
訂正後の関連テストは**100 passed（8 steps）、0 failed、2秒**。
ログは`outputs/bench/karume/2026-09-14_22-45-54_singleton-permute-tests-retry-jtavy2be/tests.log`。

試作の型検査設定、準備スクリプトの文字列一致、集計のphase名・feature列挙順・warmup JSONの形にも準備上の失敗があり、
各失敗を専用ディレクトリへ保存して新規ディレクトリで訂正した。失敗した候補を古いファイルで実行し続けていない。
参照のWGSL・golden・SHA条件や既存資産は変更していない。
最初の全体verifyは主担当が自己レビューの補足のため中断し、上記の所有権保護まで整えて再実行した。中断をテスト失敗や成功として扱わない。
全体verifyの結果は末尾に記録した。

## 次に行うこと

1. M2 Chromeでコピー削減後の動作と速度を確認する。`deno task bench:llm-browser`の既存初期選択
   （両E2B・parallel・dense・RMS融合・投入768、2設定20生成）が今回必要な構成であり、全パターンは不要。
   別日の結果との差は環境変動を含むので、そのまま厳密な改善率としない。
2. 次の実装候補は、state attentionの3 passをまとめる案とgate/upでの入力処理共有。
   現状と長文時の費用を測り、数値順が変わる場合は既存方針どおり任意指定で設計する。
   今回の計測だけで大型融合や同期方式の安全性を保証しない。
3. マージ前には今回のruntime 3ファイル・融合候補ツールとテストを追加差分として読む。
   以前のSolレビューは`315732a`までで、今回の最適化をレビュー済みとは扱わない。

## 全体検証の完了

`deno task verify`は**ok | 2961 passed (811 steps) | 0 failed | 5 ignored (2 steps) (26m11s)**で成功した（exit 0、全体1578.493秒）。
ログ・実行開始時の全追跡ファイルのSHA・差分は
`outputs/bench/karume/2026-09-14_23-22-03_singleton-permute-final-verify-8e4okmux/`。
開始から終了まで検証対象ファイルは不変。終了後は本結果の記録と、コピー対象の説明コメント1行を元の表現へ戻す訂正だけを行い、実行文・テスト・数値条件は変更していない。
5 ignoredは既存の母音検出の実音声・配布資産不足、2 ignored stepsはDenoのsubgroup機能不足。実画像fixture不足でテスト内部から省略される既存ケースもログへ残る。
M2実測・広い品質評価まで完了したことは意味しない。文書リンク・生データ56ロードのSHA・モデル計画12条件の一致も照合した。
