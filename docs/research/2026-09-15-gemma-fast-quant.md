> 2026-09-15時点のGemma E2B高速quant統合とRTXでの自動検証のスナップショット。Apple M2の既存実測とは区別する。

# 検収済み融合をE2Bの高速quantへ宣言

通常Gemma 4と固定mobile QATの新しい配布に`i4-fast`を追加し、E2BのdefaultQuantにする。
通常版は並列GEMVとRMS→add融合、QATはさらにlinear→SRQ融合を宣言する。
既存の`i4`と`i4-gemvpar`の意味・重み写像は維持し、E4Bは`i4`のまま。
変更と互換性の正本は[ADR 0104](../decisions/0104-gemma-fast-quant.md)。

## 適用と参照への戻し方

新しいhubは融合のbooleanを保存定義として読む。Gemmaの許可表と共通のSession写像を更新した。
実効値は呼び手の明示指定 → quant.session → runtime既定の順に選び、target/drafterへ渡す。
falseはquantのtrueより優先し、nullや配列を未指定へ置き換えない。未対応のquant欄は重みshard取得前に拒否する。
SRQ融合が有効なまま実効の`linearGemvReduce`がparallelでないときは拒否する。
どこにも宣言が無い場合も含むため、`quant: "i4"`を選んだまま融合だけをtrueで明示しても拒否する。
参照へ戻すには`quant: "i4"`を選ぶ。
並列GEMVだけを残すなら`quant: "i4-gemvpar"`。既定quantから個別に戻すには該当の融合をfalseにする。

既存のローカル配布と公開済みrevisionは書き換えていない。新しい出力先へ配布を組み立て、
CLIの`--source`へ渡すと新既定を選ぶ。`--quant i4`で参照経路も選択できる。
公開済み0.12.0のreaderは新欄を受理しない。公開pinやパッケージ版は今回変更していない。
比較画面は独立したA/B設定を明示するため、quantの融合設定よりその指定が優先する。
今回、比較画面の初期選択は変更せず、同じM2の40生成を再依頼しない。

## 既定化の根拠と範囲

[M2のRMS融合](2026-09-13-rms-subgroup-reduction.md#利用者のm2結果)と
[linear→SRQの40生成](2026-09-15-m2-linear-srq-adoption.md)で出力と速度を検収済み。
後者は代表値約3%改善だが標本の範囲が重なり、安定した改善率の保証とはしない。
広い品質・長文・E4Bへの適用は未検収のまま残す。

投入上限とprefillバケットは今回変更しない。投入上限1024のままでも融合の効果があることを、
RTX 3080 Ti / Chrome 153の別比較で確認する。投入768を併用した利得とは分けて評価する。

## 自動検証

配布の同一重み、宣言と明示設定の優先順位、実GPUでのカーネル適用、通常版の投機/非投機一致を検査する。
新しい配布を使うChrome・CLI、旧readerの拒否、exporter/recipeのpytest、全体verifyの結果は完了後に以下へ記録する。

## 自動検証の結果（2026-09-19）

`a287d17`と同一内容の作業ツリー（記録時点ではコミット前）を、RTX 3080 Tiで検証した。
`deno task verify`はfmt・lint・型検査を通過し、最終出力は
`ok | 2973 passed (826 steps) | 0 failed | 5 ignored (2 steps) (32m27s)`。
5 ignoredは母音検出の実音声4件と配布形1件で、いずれもローカル資産の不足。
2 ignored stepsはDenoがsubgroupに未対応なため。今回の融合宣言・Session検証にスキップは無い。

exporterとrecipeのpytestは両方を実行した。`tools/export-recipes`が2802 passed / 4 skipped、
`tools/exporter`が3227 passed / 1 skipped。

新しい配布を使うChrome・CLIの実走と、公開済み0.12.0のreaderが新欄を拒否することの実機確認は**未実施**。
この節の結果は自動検証だけを表し、上の「自動検証」節が挙げた実機確認を済ませたという意味ではない。
