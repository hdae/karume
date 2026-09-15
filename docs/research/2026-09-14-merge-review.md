> 2026-09-14時点のマージ前レビュー資料。実装の比較範囲を`4dca96a..c014231`へ固定し、後続の整理コミットは文書だけを変更する。独立レビューの結論ではない。

# codex/review-and-fix のマージ前レビュー

後続の[独立レビュー結果・利用者M2再計測](2026-09-14-merge-review-results.md)を追加した。以下はレビュー開始前に固定した範囲と検証の入口。

## 最初に確認すること

このブランチは、9/11レビューで見つかった不具合の修正、QAT mobile対応、LLMの性能改善・対話CLI・計測ツールを含む。
今回の準備では実装の追加を区切り、差分の索引、既定設定の対応、未完事項、検証の出所を整理した。
ローカルmainは`4dca96af3bee228065c9ece1b0826f23f7822f7f`で比較headの祖先。main側だけのコミットは0、本ブランチ側は62。
そのmainへはfast-forwardできる関係で、履歴の並べ替え・squash・衝突解消は行っていない。main・他ブランチ・公開資産は変更していない。
これはローカルで確認した関係であり、将来のマージ時は改めて相手のheadと比較する。

[AGENTS.md](../../AGENTS.md)、[CLAUDE.md](../../CLAUDE.md)、[ACTIVE_DESIGN](../../.claude/ACTIVE_DESIGN.md)を先に読む。
この資料と[全ファイル・62コミットの索引JSON](2026-09-14-merge-review-index.json)から担当範囲へ進む。
最初から全実測JSONを通読する必要はない。数値の根拠を調べるときに各research本文から対応JSONを開く。

```sh
git diff --stat 4dca96a c014231
git log --reverse --oneline 4dca96a..c014231
git diff 4dca96a c014231 -- packages/runtime/src
```

索引は`c014231cd4b1875870674f3bf20247a4cf7cbc41`までの356ファイル、追加99,604行・削除5,184行を記録する。
追加行のうち69,322行（約70%）は17個の実測JSON。型・WGSL・実装だけの規模と混同しない。
ファイル一覧は`git diff --no-renames --numstat`から作成し、全変更パスとの一致を確認した。
後続のこの資料・ACTIVE_DESIGN・limitations・backlog・引き継ぎの整理も、レビュー時は追加差分として読む。

## レビューの分担と読む順

以下はレビュー範囲の提案であり、まだレビューエージェントを実行した結果ではない。
各担当は現在の差分と必要な呼出元を読み、再現条件、破る契約、影響、ファイル・行番号を伴う指摘を返す。
推測は推測と明記し、実装を変更せず提出する。既知の未対応と新しい不具合を区別する。
GPU検証を並走させず、主担当が差分と根拠を確認してから修正を分割する。

| 担当                           | 主な入口                                                                                                                                                                                                | 特に確認する契約                                                                                                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. レビュー修正・生成と所有権  | `tools/exporter/src/karume/{normalize,aten_handlers}.py`、`packages/models/src/{concurrency,generation}`、`gemma/{chat-session,ple}.ts`、`irodori/pipeline.ts`、`packages/hub/src/fetch.ts`             | 記号viewの単位軸限定、全マスクsafe_softmax、未開始return/throw、doneとcleanupの順序、発話コピー、PLE破棄後の再登録、Irodori多重故障、診断callbackを除いた投機計時、samplerのNaN・同値・順序 |
| 2. GPUの安全性・数値・カーネル | `packages/runtime/src/{kernels,codegen}`、`runtime/{fusion,recipe-builder,estimate,weight-residency}.ts`、対応GPUテストとWGSL fixture                                                                   | 読む前の端チャネル保護、バリア到達、scale添字、key決定性、元の積和順、M1/4/8先頭行、subgroup機能不足の拒否、数値を変える指定と参照経路の分離                                                |
| 3. batch・実行の決着           | `packages/runtime/src/gpu/{device,pipeline-cache}.ts`、`runtime/{executor,generation-context,session-types}.ts`、`packages/models/src/gemma/greedy-output.ts`                                           | 非同期compileの失敗共有、readback予約、1フェンス、device消失、error scope、flush-before-destroy、enqueue完了とbatch最終決着、contextのadvance/poison、disposeの循環待ち                     |
| 4. INT2・固定QAT・配布設定     | `packages/runtime/src/{format,ops,reference}`、exporterの固定writer、`tools/export-recipes/gemma4_qat/`、`gemma4/distribution.py`、`packages/models/src/gemma/{qat,rope,pipeline,ple}.ts`、hub manifest | 固定整数・scaleのbyte保持、SRQ境界/特殊値、I2の範囲・整列・scale形、packed PLE schema2、通常/QAT構造門、E2B/E4B、quant優先順位、公開surface、ライセンスと未公開資産                         |
| 5. CLI・計測・導線             | `examples/shared/`、各main、`tools/{llm-speed,llm-baseline}/`、`deno.json`、README                                                                                                                      | ローカル資産の選択、KV継続/reset/中断、tokenizer固定入力、TTFTとdecodeの式、暖機、比較順、UI設定ロック・保存JSON、ローカルHTTPのパス/Range処理、計測外のDLと文章復号                        |

担当2と4は`weight-storage.ts`・`recipe-builder.ts`、担当3と4は`executor.ts`・`pipeline.ts`を共同確認する。
担当1と3は生成終了・cancel・disposeの順序を横断して確認する。名前の似たテストだけで担当を閉じない。
テスト・fixture・文書も含む全パスは索引JSONに載せ、各実装コミットから関連テストへ辿れる。

## 依存関係とコミットの見どころ

既存履歴は作業単位で分割済み。以下の矢印は意味上の依存で、個別cherry-pickの安全性を保証する表ではない。

- レビュー修正: `d4b351c` / `555f358`（exporter反例）→ `03e96c1` / `edf76f4` / `3bdaea6` / `2086d75`（所有権・解放）→ `2f85b2a` / `0d9284e` / `b054dad` / `732734f`。
  原指摘と再現・検収は[対応表](2026-09-10-codex-mtp-optimization.md#9-月-11-日レビューの対応)。[9/6と直近の棚卸し](2026-09-10-codex-mtp-optimization.md#9-月-6-日および直近レビューの棚卸し)も参照。
- 固定QAT: `ffc532b`（INT2）＋`f4752c6`（SRQ）→ `449e7e1`（固定writer）＋`fc3fd8f`（packed PLE）→ `022cb09`（recipe）→ `7ed64b3`（共通pipeline/CLI）。契約は[ADR 0097](../decisions/0097-gemma4-qat-integration.md)。
- 小出力生成: `8ec3df0` / `50675cc`（topkと非正規数）＋`f3d05bd`（一括readback）→ `d10cb43`（context予約）→ `2ea0ad5`（Gemma 8B decode）。[ADR 0054](../decisions/0054-resident-loop-and-fence.md)・[0066](../decisions/0066-generation-context-state-slots.md)・[0083](../decisions/0083-generation-api-surface.md)を合わせて読む。
- 数値を変える最適化: `9e859ec`（parallel）→ `6aae325`（quant既定）→ `7114d0c`（RMS→add）、`744cdbe`（RMS subgroup）、`17578a9`（GEMV subgroup）。M2採否とUI整理は後続コミットを含めて読む。
- 参照を保つ最適化: `f8bbaf1` / `eaccc9b`（f16/f32 GEMV）、`92d219e`（小幅RMS）、`623b8b7`（async compile）、`2e4da66` / `ffaf4b4`（行ブロック初回準備）、`a0d6d13`（PLE行cache）、`ecc3801`（BSHD RoPE）、`fbb91e8`（大語彙I8）。各コミットの検収範囲を越えて全GPUのビット一致を主張しない。
- デモ・計測: `8bdd7d4` / `3291798`（追加LLMと対話）、`d27bc99`（暖機・TTFT）、`147104a` / `cac2574` / `0adfa5a`（品質・速度・Chrome）、`e03114d` / `0ee9e48`（比較UI）。

MTPの導入、`readPleShard`から`openPleShard`への公開面変更、既存の可変capacity・共有state等は比較基点より前にある。
それらを新規差分の欠陥として再報告する前にbaseのコードと比較する。このブランチが既存契約を破っていないかの確認は必要。

## 互換性と既定設定

| 対象                   | 追加・変更                                                                                                                                    | 戻し方と境界                                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| runtimeの公開面        | `BatchScope.finishAndRead`、`EnqueueOptions.generation`、`AcquireGpuOptions.subgroups`、`LinearGemvReduce`、`RmsNormReduce`、`fuseRmsNormAdd` | 従来のfinish/runと未指定時の参照経路を保持。enqueueはエンコード完了でありcontext確定ではない                  |
| IR / safetensors       | `i2` / `I2`と`static_quantize`                                                                                                                | manifestはkarume/4のまま。旧readerは新語彙を拒否。I2は公式safetensors readerの互換形式ではない                |
| packed PLE             | schema2とI2/I4                                                                                                                                | schema1/I8は維持。旧readerはschema2を拒否                                                                     |
| quant.session          | `linearGemvReduce: sequential / parallel`                                                                                                     | 新欄は旧readerが拒否する。明示指定が優先。subgroup・RMS融合・submitPolicyは保存語彙に足していない             |
| 通常/QAT E2Bの新規配布 | `defaultQuant: i4-gemvpar`                                                                                                                    | 従来i4と重み写像は同一。`quant: i4`または`linearGemvReduce: sequential`で参照を選べる。既存資産は書き換えない |
| E4B                    | 新規配布も既定i4                                                                                                                              | E2Bの検収をE4Bの既定変更へ外挿しない                                                                          |
| QAT公開入口            | `Gemma4QatPipeline`と`@karume/models/gemma4-qat`                                                                                              | 通常Gemmaと別family。textのみ、MTPなし、公開source pinなし                                                    |
| ブラウザ比較画面       | 両E2B・parallel・dense chunk64・RMS融合・投入768、2設定20生成                                                                                 | モデル・CLIの既定とは別。sequential・subgroup・RMS・Transformers.js等の比較選択肢を維持                       |

f16/f32は**重み格納**と**計算精度**を区別する。parallelの追加はf32計算に限り、f16/a8計算との組合せは拒否する。
RMS subgroupは加算順が変わる。GEMV subgroupは既存parallelの配分・木を保つが、未検証GPUのビット一致保証ではない。
両subgroupは必要featureと32レーン既知解を要求し、黙って別経路へ戻らない。[提供制限](../limitations.md#rms融合とsubgroup最適化の提供範囲2026-09-14整理)を参照。

既存のruntime tiny goldenは変更せず、`static_quantize_block`の2ファイルを追加した。
WGSL snapshotは端チャネル保護や行ブロックの生成形変更で更新されているため、「snapshotが変わった」だけで許容誤差の緩和とは判断しない。
新しい既定quantと参照goldenは別々に検証している。参照の許容差・SHAや検査条件が弱まっていないことはレビューで確認する。

## 検証の出所と再実行

`c014231`の直前にRTX 3080 Tiで`deno task verify`を完走した。
**2,959 passed（802 steps）/ 0 failed / 5 ignored（2 steps）、25分42秒**。
`outputs/bench/karume/2026-09-14_18-33-25_i4-lane-verify-tixp3atg/`にログとソースSHAを保存した。
5 ignoredは母音検出の実音声4件と配布形1件の資産不足。2 ignored stepsはDenoのsubgroup機能不足。
BiRefNet/Lucida 2048の実画像fixture不足を理由に、テスト内部で実行されないケースもログへ明示されている。
全体の緑を、全モデル・全資産・全GPUの検収と呼ばない。

マージ準備でPythonも再実行した。`outputs/bench/karume/2026-09-14_19-01-32_merge-pytest-xcoxqj9a/`:

- exporter: **3,227 passed / 1 skipped / 332 warnings、47.92秒**。
- export-recipes: **2,800 passed / 4 skipped / 147 warnings、246.08秒**。
- `tools/llm-baseline`と`tools/llm-speed/test_timing.py`: **22 passed、4.11秒**。
  ログは`outputs/bench/karume/2026-09-14_19-03-37_merge-benchmark-tests-qxynn32h/`。

Pythonのskip・warningは各ログを参照する。既存アセットを再変換して合格させたわけではない。
文書整理後の全体verify結果は本資料の末尾に記録した。

```sh
deno task verify
```

Pythonはそれぞれの作業ディレクトリで`uv run pytest`を使う。ベンチツールの検査はリポ直下から次を実行する。

```sh
PYTHONPATH=tools/export-recipes tools/.venv/bin/python -m pytest -q tools/llm-baseline tools/llm-speed/test_timing.py
```

Chromeの実装検収とM2結果は[並列GEMV](2026-09-13-gemv-subgroup.md)、[M2の採否](2026-09-13-m2-gemv-subgroup-adoption.md)、[RMS](2026-09-13-rms-subgroup-reduction.md)へ辿る。
今回の文書整理ではChromeの生成を取り直していない。最新の実験は[大きいI4の不採用比較](2026-09-14-i4-lane-comparison.md)。
計測を再開する場合は`deno task bench:llm-browser`と[ブラウザ手順](../../tools/llm-speed/browser/README.md)を使い、新しい実験ディレクトリへ保存する。

## マージと公開の違い・未完項目

コードをマージしても、`models/`・`outputs/`の変換済みモデルや実験台本はgitへ含まれない。
`.claude/reviews/`の原レビューも追跡外で、移植可能な入口は本資料・追跡済みresearch・テスト・CLIとする。
実測JSONに記録したローカル生データのパスは、別checkoutに自動では現れない。既存の量子化モデルや実測値を捏造して補わない。
QATの配布作成は[recipe README](../../tools/export-recipes/gemma4_qat/README.md)、追加LLMは各[MiniCPM5](../../examples/minicpm5/README.md) / [Qwen3](../../examples/qwen3/README.md)の手順を読む。

公開source表・既存の公開pinはこのブランチで変更していない。パッケージのバージョン更新・release・HF uploadも含めない。
新しいIR/PLE/quant語彙を使う配布形の公開には、対応readerとの組合せを[release-runbook](../release-runbook.md)で検収する必要がある。
QATのライセンス・改変通知は[THIRD_PARTY_NOTICES](../../tools/export-recipes/gemma4_qat/THIRD_PARTY_NOTICES.md)とrecipeの同梱処理を確認する。
汎用exporter coreへのモデル固有依存や上流ソースの混入は、[ADR 0065](../decisions/0065-exporter-core-recipe-split.md)の境界で検査する。

未完は[backlog](../backlog.md)を正本とする。今回のマージ資料準備で完了扱いにしないものは以下。

- M2のGPU費用帰属、RMS→add＋投入768のモデル既定への組み込み。
- Gemma/QATの広い品質・長文・E4Bへの追加最適化、一般sampling・投機生成の転送削減。
- MiniCPM5/Qwen3の公開pipelineと長文脈、QATの公開source pin。
- exporterの同一finalへの複数writer、偽HFテストURLのcache分離等の[既知問題](../known-issues.md)。
- Wan / MiniMax H3のブラウザ実装。現在は事前調査のみ。

M2のGEMV subgroupの80生成は完了して採用見送り。RMS subgroupも既定には採用しない。
大きいI4のL8案は遅化のため不採用。これらは「未実装だから再度実装する候補」として扱わない。
次は上記の分担による独立レビューを受け、根拠のある指摘を作業単位で修正・検証・コミットする。

## 文書整理後の最終検証

`deno task verify`は`ok | 2959 passed (802 steps) | 0 failed | 5 ignored (2 steps) (25m46s)`で完了した（exit 0、全体1547.670秒）。
ログは`outputs/bench/karume/2026-09-14_19-10-32_merge-docs-verify-gixbe167/verify.log`。検証開始時からTS/WGSL/HTML 782ファイルのSHAは不変。
348ローカルリンク、全変更パスと索引、数値、mainとの祖先関係、既存tiny goldenと公開source表の非変更を照合した。
終了後の変更はこの検証記録の追記とfmtだけで、推論コードやテストを変更していない。

## 独立レビュー後の追加差分（2026-09-14、基準c265a60）

添付資料を現行コードで再検証し、[要素順を保つpermuteのコピー削減](2026-09-14-reference-rope-optimization.md)を追加した。
この差分は以前のSolレビュー（315732aまで）の範囲に含まれない。固定索引は当時の範囲のまま保持する。

追加で読む実装は`packages/runtime/src/runtime/fusion.ts`の`planAliases`と
`packages/runtime/src/runtime/recipe-builder.ts`のpermute分岐、`estimate.ts`と融合候補ツールの共有判定への接続。
非単位軸の順序判定、入力・重みからのコピーを残してcopyOutputsの既存受理条件を保つこと、
別名時にコピーを発行しないこと、既存retain/release・出力pinとの整合、
実行とメモリ見積りが同じ述語を使うことを確認する。
回帰テストは`runtime_permute_alias_test.ts`、`gpu_permute_alias_test.ts`、独立添字oracleの`helpers/permute.ts`。
IR・公開設定・dtype対応・数値演算・goldenは変更しない。理由と境界は[ADR 0011](../decisions/0011-layout-strategy.md#要素順を保つpermute2026-09-14)。

RTXの最終実装160生成は通常約2.44%、QAT約0.93%速いが、別走行のQATは逆転しており、固定改善率の保証ではない。
M2は未検収。588生成の出力比較・関連125テスト・全体verifyの出所は上記調査記録に集約する。
外部カーネルは読解のみで複製していない。RMS→RoPEの追加融合は試作に留め、製品へ入れていない。

## attention融合の追加差分（2026-09-15、基準ff6da66）

[調査と検証](2026-09-15-attention-fusion.md)と[ADR 0102](../decisions/0102-state-attention-stats-pv-fusion.md)を追加。
runtimeの任意指定`parallel-fused`、新しい行統計/PV融合カーネル、実行・メモリ見積りの共通適用判定を読む。
共有メモリの最大値→分母→PV部分和の再利用のbarrier、NaN・空行・pad、ring、
M≤8/列上限≤1024の適用範囲、readonly非適用、statsの確保/解放削除と見積りの一致が確認点。
GemmaはSession構築時の同じ設定を見積りへ渡す。参照WGSL/goldenとquant/defaultQuantは維持する。

回帰は`gpu_state_attention_fused_test.ts`、`gpu_state_execution_test.ts`、plan/estimate/snapshot、
実モデルMTPのverify行0/decode比較。Chromeの比較画面は8設定80生成を初期選択にした。
この差分も315732aまでの独立レビューには含まれない。M2のattention採否と保留候補の組合せは未完。
前節のpermuteについては最新M220生成で出力一致を確認済み。

## 保留候補の併用調査の追加記録（2026-09-15、基準b9d41ef）

[調査記録](2026-09-15-held-combinations.md)と[数値・raw hash](2026-09-15-held-combinations-results.json)を追加した。
この追加単位は文書と調査結果のみで、実験用の融合・重み配置・投入設定を製品コードへ移していない。
920生成の探索/再測定、35生成のGPU診断、元の非融合SRQを使う360条件の数値比較を区別して読む。
前節で未完だった保留候補の併用測定は完了した。M2のattention検収と、QATで改善したlinear→SRQの製品統合は未完。
マージ時にはattentionの追加差分も引き続き確認する。315732aまでの独立レビューで確認済みとは扱わない。

## 実行設定の入力型検査（2026-09-15、基準62675e8）

[修正記録](2026-09-15-execution-option-validation.md)を追加した。
新しいattention見積り設定と、既存Sessionの実行設定の型を、Object.hasOwnのキー変換より前に検査する。
確認点は配列などの拒否、診断で利用者の変換を呼ばないこと、文字列の既存エラーと有効な実行設定を保つこと。
カーネルと既定は変更しない。この修正も以前のSolレビューの範囲には含まれない。

## linear→SRQ融合の追加差分（2026-09-15、基準f6ccf44）

[実装の検収](2026-09-15-linear-static-quantize-fusion.md)と[ADR 0103](../decisions/0103-linear-static-quantize-fusion.md)を参照する。
旧レビューの範囲には含まれない。主な確認点は、常駐格納の純粋な記述とscaleの借用の分離、
元のparallel/SRQとの算術・丸め障壁、M1..8とprivate/隣接/同形状の適格条件、sharedWeightsの解放順。
新しい診断カウンタと任意のboolean設定を加え、モデル・quantの既定は変更していない。
M2のattention追試は利得が無く、画面の次比較は従来attentionを使うQATのlinear→SRQ40生成とした。

linear→SRQ製品化の全体verifyとソース不変の照合も完了。
最終件数と生ログは[検収記録](2026-09-15-linear-static-quantize-fusion.md#全体検証)に集約した。
M2の新しい融合の検収は未完で、モデル既定の変更は次の判断に残す。
