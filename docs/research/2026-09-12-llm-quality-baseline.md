# PyTorch / Transformers の LLM 品質ベースライン

> 2026-09-12時点のローカル重み・固定した小規模評価集合の記録。公開ベンチマークの完全走行や、モデル全般の品質保証ではない。

## 目的と評価条件

デモのTTFT（最初のトークンまでの時間）分離と暖機に続き、今後の高速化で品質差を測るためのCPU参照を作る。
実装は [tools/llm-baseline](../../tools/llm-baseline/README.md)。PyTorch 2.13.0+cpu /
Transformers 5.14.1 / Python 3.14.6、CPU 4 threads・float32・eager attention・cache無しで測る。
ここでのCPU所要時間は推論速度の比較に使わない。KarumeのWebGPU実行スコアはこの表に含まない。

- ARC-Easy testの先頭64問。全6モデルの選択肢が128 token以内に収まる問題を、推論前に選ぶ。
  今回は除外0問、最大75 token。`Question: {question}\nAnswer: {choice}`という同じ形式で、
  選択肢の継続部分の対数確率を合計する。別列でtoken数による平均も示す。
  chat template、例示、思考文生成、回答の文字列解析は使わない。
- WikiText-2 raw testの行を改行2個で繋ぎ、先頭8,192 Unicode文字を固定する。
  window128・stride64、各窓の位置は0から、先頭以外の各tokenをちょうど一度採点する。
  tokenizerがBOSを持つ場合は先頭へ一度だけ追加する。
  平均負対数尤度（NLL）の指数をperplexity（PPL）とする。小さいほどこの文章の予測が良い。
- PPLの比較は同じtokenizer・文章・窓条件の組に限る。Gemmaは1,870 token、MiniCPMは1,820、
  Qwenは1,922を採点する。異なるtokenizer間のPPLをモデル順位として扱わない。
- ARCの95%区間はWilson法。64問の小標本であり、数問の差から優劣を断定しない。
  先頭からの連続した問題であり、無作為抽出した全課題の推定とは扱わない。
  日本語、長文、多ターン、自由生成の品質評価は別途必要。

## 重みの条件と参照の独立性

通常Gemma E2B/E4B、MiniCPM5-2B、Qwen3-0.6Bは元checkpointと保存済み量子化重みを各1回測る。
保存済み整数とscaleを逆量子化して公式モデルへ戻し、再校正・再量子化はしない。
MiniCPM/Qwenは既存のGPTQ候補。通常4構成の主な線形投影はINT4・group32、embeddingと出力headはINT8、
正規化などはfloat32で保存されている。通常GemmaのPLEはINT8で、全量をfloat32化せず必要な行だけ同じ式で読む。
保存dtypeごとの要素数は生データの`quantization-census.json`に記録した。
E4Bの配布形、MiniCPM/Qwenのseriesはローカル実験資産であり、公開済みと扱わない。

QAT E2B/E4Bは公式checkpoint自体が固定量子化済みのため、各1回のみ。
通常版とは別checkpointなので、両者のスコア差を保存時の量子化だけの影響とはみなさない。
保存グラフの固定整数・scaleの値、PLEの全バイト、線形演算前後の正のSRQ（固定scaleによる活性値の丸め）を照合し、
公式Transformersの量子化モジュールを使う。E2Bは539 initializer・550 SRQ scale・5 PLE shard、
E4Bは664 initializer・684 SRQ scale・3 PLE shardを照合した。

KarumeのGPU executorは参照値の生成に使わない。元モデルのforward実装はコピーせず、公式ライブラリから読み込む。
量子化条件の一致は、CPU/GPUの縮約順序やQAT丸め境界を含むビット一致の主張ではない。
通常Gemmaの行単位PLE読取は、元BF16行・保存I8行・layer scale・重複token・shard境界のテストを通す。
Qwen/MiniCPMの保存済み重みは既存の参照fixtureとも照合し、最大絶対差はそれぞれ
2.575e-5 / 4.387e-5、既存と同じ許容差1e-3以内だった。期待値と許容差は変更していない。

## データと再現

生データの根は `outputs/bench/karume/2026-09-12_llm-quality-baseline-tutu2zk0/`。
`data/sources.json`にdataset revision・元Parquet・JSON・dataset cardのSHA-256、
`suite/suite.json`に選定問題・全入力token ID・tokenizerの指紋を保存する。
各`full-<model>-<source|stored>/`に、重み・config・tokenizer・評価コードの指紋、
問題別・窓別のNLL、実行環境と集計を残す。途中失敗と少数問題のsmokeは別ディレクトリであり、本表に混ぜない。

[ARCの公開データ](https://huggingface.co/datasets/allenai/ai2_arc)はCC BY-SA 4.0。
[WikiTextの公開データ](https://huggingface.co/datasets/Salesforce/wikitext)はmetadataでCC BY-SA 3.0 / GFDL、
card本文でCC BY-SA 4.0を参照しているため、固定revisionのcardも保存した。
データ本文はgit管理外の出力に置く。窓を重ねる評価方針は
[Transformersのperplexityガイド](https://huggingface.co/docs/transformers/en/perplexity)を参照し、
重複範囲と次tokenへのshiftを明示的に数える実装を独自に書いた。

## 結果

利用者が主目的をtok/s比較と明確化したため、完了した4構成を品質参考値として保存し、速度評価を先行する。
Gemma通常/QATのE2B/E4Bは読み込み・少数問題の採点まで成功。本評価の正答率はまだ記録しない。
途中の通常E2B source走行も保存し、完走結果へ混ぜない。再開は新しい出力先を使う。

| モデル      | 重み   | ARC 正答数 | token平均で選択 |    PPL |
| ----------- | ------ | ---------: | --------------: | -----: |
| qwen3-06b   | source |      42/64 |           43/64 | 36.028 |
| qwen3-06b   | stored |      36/64 |           37/64 | 48.019 |
| minicpm5-2b | source |      47/64 |           48/64 | 25.888 |
| minicpm5-2b | stored |      41/64 |           45/64 | 29.001 |

`source`は元checkpointの値、`stored`は保存済み整数・scaleを戻した値。計算はいずれもCPU float32。
全問題の選択肢NLL、窓の各token NLLから独立に再集計し、保存summaryと一致した。
全4構成の入力集合・コード指紋も一致する。生検収は`aggregate-completed.py`と同名のlog/JSON。
[追跡する集計JSON](2026-09-12-llm-quality-results.json)に元ログの指紋とWilson区間を残す。

この集合では保存済み重みでPPLが上がり、ARCの正答数が減った。Qwenは正解から不正解へ9問、
逆方向へ3問変わっている。これは保存形全体の差であり、INT4の線形重みだけの効果とは断定しない。
以前の短文でのGPTQ改善を撤回する結果でもない。元重みとの品質差が残ることを別の評価で記録した。
品質の既定設定や許容差は変更せず、広い評価・校正改善の比較基準にする。

## 検証と残件

Python単体検証は18 passed / 0 failed（`unit-final.log`）。Ruffの検査・整形確認も通過。
本採点前のパス結合の誤りはローダー側を修正し、新規出力先でGemma4構成を再検証した。
固定丸めの照合と既存fixtureの数値条件は弱めていない。

優先する残件はPyTorchとDeno/WebGPUのTTFT・decode tok/s比較。品質側はGemma4構成の本採点、
次にKarumeへ同じ入力を与えた比較、日本語・長文・多ターン評価を残す。
全体検証は **2,934 passed（760 steps）/ 0 failed / 5 ignored、25分6秒**。
ログは`outputs/bench/karume/2026-09-12_llm-quality-verify-C2wXeG/verify.log`。
GPUの別計測と並走させず、検証中に品質評価のPythonコードは変更していない。
