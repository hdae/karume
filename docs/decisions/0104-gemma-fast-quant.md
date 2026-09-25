# 0104: 検収済み融合をGemma E2Bの高速quantに宣言する

- Status: accepted（2026-09-15、利用者の既定化継続指示）。自動検証の結果は[統合の検収](../research/2026-09-15-gemma-fast-quant.md)に記録。広い品質・長文・E4Bは未検収。
- 関連: [0038](0038-manifest-v1.md)、[0058](0058-numerics-opt-in-contract.md)、[0098](0098-linear-gemv-parallel.md)、[0099](0099-rms-norm-add-fusion.md)、[0103](0103-linear-static-quantize-fusion.md)。
- 統合の検収: [新しいquantの配布・実行](../research/2026-09-15-gemma-fast-quant.md)。
- M2の根拠: [RMS融合](../research/2026-09-13-rms-subgroup-reduction.md#利用者のm2結果)、[linear→SRQ](../research/2026-09-15-m2-linear-srq-adoption.md)。

## 変更する定義

通常Gemma 4と固定mobile QATのE2Bに、新しい`i4-fast`を追加してdefaultQuantに選ぶ。
参照用`i4`と並列GEMVだけの`i4-gemvpar`は同じ意味で保持する。
3種類とも重み写像は同じで、再量子化・重みコピーは要らない。

| E2B  | i4-fastのsession宣言                                                                                                                                                                                 |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 通常 | linearGemvReduce: parallel、fuseRmsNormAdd: true                                                                                                                                                     |
| QAT  | 上記＋fuseLinearStaticQuantize: true＋packedStaticQuantize: true（[0105 追記 2](0105-packed-static-quantize-activations.md#追記-22026-09-20-語彙への昇格と-i4-fast-の宣言段-1b-の棄却)・2026-09-20） |

検収範囲外のE4Bはi4だけを維持する。runtimeの既定、参照WGSL/golden、公開pinを変更しない。
新しい配布をrecipeから組み立てたときに既定が変わり、既存のローカル配布や公開済みmanifestを自動で書き換えない。

## 保存語彙と互換性

hubのSessionSpecへ省略可能な2つのbooleanを追加する（3つ目の`packedStaticQuantize`は[0105 追記 2](0105-packed-static-quantize-activations.md#追記-22026-09-20-語彙への昇格と-i4-fast-の宣言段-1b-の棄却)が同じ流儀で加えた）。true/falseだけを許し、null・数値・文字列を拒否する。
falseを省略や欠如へ変換せず、modelsの共通写像も同じ欄へ明示して転送する。
manifestはkarume/4のまま。新readerは旧manifestを同じ意味で読み、旧readerは新しい融合欄を未知キーとして拒否する。
新しい配布には対応するhub/modelsが必要で、公開済み0.12.0で読めるとは扱わない。

Gemmaがquant.sessionから受理するのはlinearGemvReduceと融合欄（`fuseRmsNormAdd` / `fuseLinearStaticQuantize` / `packedStaticQuantize`）だけ。
重みshard取得前のadmissionで、受理する欄と実効設定を検査する。
利用者の明示指定 → quant宣言 → runtime参照既定の順で各キーを選び、target/drafterの両方へ渡す。
未対応の宣言を上書きで隠さず、null等の不正な明示値をquantの値へ戻さない。

SRQ融合が有効なまま実効のlinearGemvReduceがparallelでないときは拒否する。
sequentialやparallel-subgroup32の明示に限らず、どこにも宣言が無い場合（i4を選んで融合だけをtrueにする場合）も拒否する。
その比較ではfuseLinearStaticQuantize: falseとpackedStaticQuantize: falseも指定するか、i4/i4-gemvparを選ぶ（packedも同じparallel必須の拒否を持つ）。
型の正しいfalseは常にquantのtrueより優先する。fromAssetsはquant選択が無いので従来どおり明示したオプションだけを使う。

## ホスト側の方針を分ける

投入上限とprefillバケットはホスト側の方針であり、quant.sessionへ保存しない。
モデル/GPU名から融合や並列化を追加せず、今回の既定化はquantの宣言だけで選ぶ。
投入上限1024のままでも融合単独の全体速度を比較し、768の併用による利得とは分けて評価する。
現在のCLI・モデルの投入政策とバケットはこの変更で切り替えない。

## 検収項目

Status: acceptedは決定そのものの承認状態であり、以下の項目を実施済みという意味ではない。実施結果は[統合の検収](../research/2026-09-15-gemma-fast-quant.md)が持つ。

- 保存語彙の欠如/true/false/不正値、共通写像、宣言と明示指定の優先順位、未対応の組合せを検査する。
- 配布recipeのi4とi4-gemvparの保持、i4-fastの重み写像・既定選択、E4B非適用を検査する。
- 実モデルで既定/参照/明示有効/明示無効を実行し、カーネルの適用と生成一致を確認する。
- 通常版のtarget/drafterへの伝播と同一設定での投機検証を確認する。QATのMTP対応は追加しない。
- 新しく生成したquant宣言を使うChrome比較、deno task verify、exporter/recipe両方のpytestを通す。

## 追記（2026-09-25）— コンテナ後の manifest

「manifest は karume/4 のまま」は、ADR [0109](0109-manifest-v5-container.md) で `karume/5` へ進んだ後も
`quants[].session` の欄がそのまま引き継がれている（0109 決定 2）。「重み shard 取得前の admission」の
shard は、ADR [0108](0108-container-format.md) 以降はコンテナの part を指す。格納の正本は codec / layout
（0108 決定 12 / 13・[container-v1](../container-v1.md) §6）。
