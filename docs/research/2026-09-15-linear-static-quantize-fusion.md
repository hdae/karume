> 2026-09-15時点の実装・実測スナップショット。利用者のApple M2計測と、RTX 3080 Ti / Chrome 153の自動検証を区別する。

# M2 attention追試とlinear→SRQ融合の製品化

[前回の併用実験](2026-09-15-held-combinations.md)から、改善の大きかったlinear→static_quantizeだけを独立した任意設定へ整理した。
基準は`f6ccf44`。数値・入力と生データのhashは[結果JSON](2026-09-15-linear-static-quantize-results.json)、仕様と所有権は[ADR 0103](../decisions/0103-linear-static-quantize-fusion.md)を正本とする。
外部のソースコードは複製していない。

## 利用者のM2 attention結果

添付`gemma-browser-2026-09-15T04-06-55.199Z.json`は同一のcheckoutとbundleで、通常/QATの各モデルを
parallel→parallel-fused→parallel-fused→parallelの順に比較した80生成。全てtoken/stop/textが一致し、失敗は0件。
同じモデルの資産、入力、量子化、並列GEMV、dense chunk64、capacity128、RMS→add融合、投入上限768を照合した。
初回・追加暖機を除いた3回の中央値を入力・ロードごとに求め、さらにその中央値で比較した。

| M2 / Chrome | parallel tok/s | parallel-fused tok/s |     差 |
| ----------- | -------------: | -------------------: | -----: |
| 通常E2B     |        33.3776 |              33.0831 | −0.88% |
| QAT E2B     |        34.8201 |              34.7632 | −0.16% |

TTFTの中央値は通常374.11→373.93 ms、QAT432.66→432.98 msでほぼ不変。
今回のM2では高速化の根拠にならず、attention融合は任意指定に残す。次の比較は従来のparallelを基準にする。
RTXでの小幅な改善をM2へ一般化せず、モデル既定へは昇格しない。
原本と設定の照合は`outputs/bench/karume/2026-09-15_04-08-31_m2-attention-validation-d2cyaal8/`に保存した。

## 実装した範囲

- `fuseLinearStaticQuantize: true`でのみ融合する。`linearGemvReduce: "parallel"`とf32演算が必要。
  真偽値以外と未対応の計算方式は構築時に拒否する。モデルやGPU名から並列設定を追加しない。
- private・隣接・同形状のlinear→SRQ、検証済みの常駐INT2/4/8・M1..8だけに適用する。
  実配布のQAT E2BはM1/4/8で275箇所、M32/40/64と通常E2Bは0箇所。
- 元のparallelの加算とSRQの整数表を共用し、最後の出力だけを融合する。
  実験用のWGSL文字列置換を残さず、型のあるgeneratorで生成する。融合内の境界探索だけを固定回数へ展開した。
  uniformの0とのXORを丸め障壁として維持する。
- 融合計画は常駐格納の記述だけを受け、レシピ導出時に既存のweight scaleを借りる。
  重み転置、重み・scaleの複製、別の所有者、独自の解放簿記は追加しない。
- Gemma通常/QATのpipelineへ設定を通す。IR・manifest・quant/defaultQuant・CLIの既定は不変。
  参照goldenと独立SRQも変更していない。

## RTX / Chromeの全体速度

製品コードだけをbundleし、通常E2BはABBA、QAT E2BはABBA+BAABで計120生成。
前回と同じ英語・日本語の2入力、64 token、温度0、初回と追加暖機を除いて本計測3回の中央値。
M2の採否に従いattentionはparallel。それ以外は並列GEMV・dense chunk64・capacity128・RMS融合・投入768を揃えた。

| RTX / Chrome         | 非融合 tok/s | 融合指定 tok/s |         差 |
| -------------------- | -----------: | -------------: | ---------: |
| 通常E2B（適用0箇所） |     114.8171 |       115.4911 |     +0.59% |
| QAT E2B              |      94.1002 |        99.7389 | **+5.99%** |

QATの入力・ロード別中央値は非融合93.6803〜95.9781、融合99.1501〜100.2227 tok/s。
TTFTは39.55→39.35 msでほぼ不変。120生成のtoken/stop/textは同じモデル・入力の基準と一致した。
通常版は適用箇所0なので、約0.6%の差をこの融合の利得とは解釈しない。

各ロード前に75℃以下かつsoftware thermal slowdownが無効であることを確認し、500ms間隔で記録した。
全396標本のうち本計測は96標本。QATの68標本は71〜82℃、software/hardware thermal slowdownとも0件。
通常版の28標本ではsoftwareが1件有効だった（最初の非融合・英語・本計測3回目）。
その標本を除外せず残し、通常版の差を採用根拠にしない。500ms未満の状態を否定するものではない。
クロックやファン等のハードウェア設定は変更していない。

生データ・bundle・準備と実行スクリプトは`outputs/bench/karume/2026-09-15_04-22-46_linear-srq-product-whole-92vof_4r/`。
前回試作の約5.5%は参考であり、異なる走行の絶対値を直接引いて改善量を求めていない。

## GPU時間の帰属

QATをABBAでロードし、各3暖機＋通常生成／native pass timestamp／通常生成の計24生成。
GPU queryのresolve/mapは生成の計時後。通常のpass境界を維持し、dispatchごとの分割profileは使っていない。
全24生成が前節の基準と一致した。

| 構成・順序 | 起動数/token | GPU ms/tokenの中央値 | 融合数/token |
| ---------- | -----------: | -------------------: | -----------: |
| 非融合1    |         1411 |             7.351296 |            0 |
| 融合1      |         1136 |             6.774784 |          275 |
| 融合2      |         1136 |             6.813696 |          275 |
| 非融合2    |         1411 |             7.394304 |            0 |

GPU時間の中央値の代表値は7.372800→6.794240 ms。275 dispatchの削減と処理時間の減少を確認した。
この差をCPU単独の費用とは見なさず、生成速度の正本は前節の通常計測とする。

初版の集計は1生成を192 passと固定して193 passの走行を拒否した。
実データでは、時間予算による投入境界の違いで非融合のdecode1回だけ4 pass、残りは3 passだった。
集計を実際のembedding開始とtopk終了で分け、63 decodeそれぞれの起動数を独立に照合した。
データや性能標本を削除・修正していない。
保存先は`outputs/bench/karume/2026-09-15_04-39-41_linear-srq-product-profile-xltwfot8/`。

## 数値・寿命・画面の検証

- 既存の逐次・parallel・subgroup generatorを変更前の実装と86条件で照合し、WGSLがバイト同一。
  既存スナップショットは不変。融合版の新規3スナップショットを追加した。
- カーネルは540条件で元のparallel＋元の二分探索SRQと比較。
  INT2/4/8、短いK/1536、L2/4/8/16/32、M1/4/8、全128閾値の前/一致/後×正負、広い値、非有限を含む。
  有限はu32、NaNは分類で一致。scale0の恒等と、uniformのXORを符号反転へ変えた場合の実行も検査する。
- SessionではM1/4/8の非融合・行0一致、M9の非適用、計画再利用、後続の入力消費を確認した。
  INT2/4/8の共有scaleについて、借り手がいる間の貸し手dispose拒否、借り手破棄後の貸し手再利用を検査した。
- 関連7ファイルの検証は117 passed（7 steps）/ 0 failed。テスト準備中の型・shared宣言・
  ルール先頭一覧の不足は修正して再検証した。製品の検証条件を緩めて通していない。
- 実Chromeの画面初期選択40生成、選択ロック、設定ラベル、生成一致、JSON保存一致、
  不正URLの資産取得前拒否を検査した。初回adapter取得に失敗し、既存の上限2回の初期化で2回目にNVIDIA実GPUを取得した。
  ソフトウェアGPUへ切り替えていない。画面の検査結果は`outputs/bench/karume/2026-09-15_04-38-45_linear-srq-browser-ui-eudk359t/`。

実装と単体検証の記録は`outputs/bench/karume/2026-09-15_04-10-40_linear-srq-integration-36_zacop/`。
通常生成120＋画面40＋GPU診断24の計184生成がRTXで基準と一致した。
これは今回の短文入力での一致であり、長文・広い品質評価やM2の融合結果を保証するものではない。

## モデルと投機デコードの回帰検証

`e2e_gemma4_quant_test.ts`は通常/QATの2件・2 stepsが成功。
quantの既定・参照・明示上書きの関係を保ち、新フラグがQATの実カーネルへ届くことと生成一致を検証した。
`e2e_gemma4_speculative_test.ts`の投機⑤は1件・8 stepsが成功。
新フラグを含む通常版のverify行0とdecodeについて、全262,144語のu32一致を確認した。
通常版のSRQ融合数は0であり、QATのMTP対応を検収したという意味ではない。
ログは`outputs/bench/karume/2026-09-15_04-43-08_linear-srq-model-tests-5zwsr8ql/`。

## 全体検証

`deno task verify`はfmt・lint・型検査を通過し、テスト走行を完了した。
最終出力は`ok | 2970 passed (823 steps) | 0 failed | 5 ignored (2 steps) (27m4s)`、コマンド全体は1630.7秒。
スキップは実音声WAV不足による母音検出4件、配布資産不足の母音検出1件、
Deno未対応のsubgroup実走2 steps。今回の追加カーネル・Session検証にスキップは無い。
RTX 3080 TiでGPUテストを実行し、GPUベンチとは並走させていない。
実行前後のTS/WGSL/HTML 798ファイルのSHAは同一。既存golden・参照スナップショットの変更は無い。
ログと検証対象hashは`outputs/bench/karume/2026-09-15_04-50-17_linear-srq-full-verify-hroiuby3/`に保存した。
検証後の変更は結果の記録と文書の整形だけで、推論コード・テスト・比較画面を変更していない。

## 次の検収

M2では`deno task bench:llm-browser`を起動し、開始ボタンだけでQATの非融合→融合→融合→非融合を40生成比較できる。
通常版や過去の候補、attention融合、参照設定も選択肢に残している。
M2結果の確認後にquant既定への採用を判断する。広い併用の既定化、E4B・他モデル・長文の検収は別の作業。
次の新候補はgate/up入力共有などだが、今回の独立融合のM2検収を先に行う。
