> 2026-09-15時点の不具合調査・修正記録。性能候補の採否とは別の、JavaScript入力境界の修正。

# 実行設定の型をキー変換の前に検査する

attention融合の追加確認で、`EstimateOptions.stateAttentionReduce`へ`["parallel-fused"]`を渡しても
見積りが成功することを再現した。文字列ではないので実際の経路判定の厳密比較には一致せず、
有効な選択として検査を通った後に別の経路を使う可能性がある。

原因は受理集合への`Object.hasOwn`の前に型を検査していなかったこと。
配列やオブジェクトがキーへ変換され、受理集合内の名前になると通過する。
新しい見積り設定に加え、既存のSessionの実行設定5項目にも同じパターンがあった。
変更前の2つの回帰テストで「拒否されるべき値が成功する」ことを確認した。

## 修正と範囲

- Sessionの`linearCompute`、`attentionCompute`、`attentionScoreStorage`、
  `stateAttentionReduce`、`linearGemvReduce`と、見積りの`stateAttentionReduce`を対象とする。
- 既定値適用後の値が文字列か確認してから受理集合を引く。
  不正な配列・オブジェクト・boolean・number・bigint・symbolは`ExecutionError`で拒否する。
- 不正な非文字列の診断には型名を使い、利用者の`Symbol.toPrimitive`や`toJSON`を呼ばない。
  bigintのJSON化による別の例外にもならない。文字列の既存エラーメッセージと受理集合は保つ。
- 有効な設定の意味、既定値、演算・WGSL・数値許容差・goldenは変更しない。
  null / undefinedに対する既存の既定代入も変更していない。新しい公開設定や保存形式の追加ではない。

## 検証

CPUの最小再現は`outputs/bench/karume/2026-09-15_02-56-33_execution-option-type-audit-nzh00i1q/repro.ts`と`before.json`。
有効な文字列は受理、配列と変換可能なオブジェクトも誤って受理、不正な文字列は拒否された。

回帰テストを先に加え、修正前の2テストがそれぞれ期待したassertionで失敗することを確認した。
`outputs/bench/karume/2026-09-15_03-15-26_option-types-regression-before-kgk4dtgc/`のruntime.logは「Expected function to reject」、
estimate.logは「Expected function to throw」。コンパイル失敗を再現成功として数えていない。

修正後の`runtime_executor_test.ts`と`estimate_test.ts`は**93 passed / 0 failed**。
`outputs/bench/karume/2026-09-15_03-16-09_option-types-targeted-50q6_82q/tests.log`に保存。配列など6種類×5設定と見積り6種類を検査し、利用者の変換の呼出し回数も0であることを確かめた。
文字列の既存診断・正常な構築、メモリの計算・Sessionの寿命の既存検証も含む。

## 全体検証

`deno task verify`は**2,967 passed（815 steps）/ 0 failed / 5 ignored（2 steps）**で完了した。
最終行は`ok | 2967 passed (815 steps) | 0 failed | 5 ignored (2 steps) (26m16s)`、全体1580.246秒、exit 0。
ログは`outputs/bench/karume/2026-09-15_03-18-08_option-types-full-verify-07fuvo4f/verify.log`。
検証中のTS/WGSL/HTML 791ファイルはSHA不変。カーネル・fixtureの差分が無いこと、97ローカルリンクと再現・回帰ログのSHAを照合した。
検証後の変更はこの結果の追記と文書整形だけで、推論コード・テストは変更していない。
exporter/recipeの変更はなく、この修正でPythonの再検証は対象外。
