> 2026-09-12時点のRTX 3080 Tiでの実測スナップショット。M2の結果や各フレームワークの性能上限ではない。

# LLM の TTFT・生成速度ベースライン

利用者の主目的であるtok/s比較のため、通常Gemma 4 E2B/E4B、mobile QAT E2B/E4B、
MiniCPM5-2B、Qwen3-0.6Bを同じ固定入力でDeno/WebGPUと公式PyTorch/Transformersに通した。
[品質参考値](2026-09-12-llm-quality-baseline.md)は副指標として保持し、CPU採点の所要時間を速度比較へ流用しない。
デモの暖機と表示変更は[別の検収記録](2026-09-10-codex-mtp-optimization.md#デモのttft分離と起動時ウォームアップ2026-09-12)を参照。

## 条件と再現方法

ツールは[tools/llm-speed](../../tools/llm-speed/README.md)。固定入力を1モデルずつ準備し、
保存先を変えてDenoとPyTorchを逐次実行する。資産はローカル直読で変更・再量子化しない。
`--checkpoint`で公式config/tokenizer、`--source`で変換済み配布形/系列を指定できる。
公式会話テンプレートが無い場合は`--chat-template`で明示する。検証したE4Bのローカル配置ではこれが必要だった。

- 実機: NVIDIA GeForce RTX 3080 Ti、VRAM 12,488,474,624 B、driver 610.57.04。
  Deno 2.9.6 / WebGPU、Python 3.14.6、PyTorch 2.13.0+cu130 / CUDA 13.0、Transformers 5.14.1。
- バッチ1、greedy、投機なし、容量128、最大64token。英語と日本語の箇条書き生成を各1件。
  公式chat templateで固定したtoken IDと停止集合を両エンジンに渡す。
  Deno側のtokenizerでも同じ入力列になることを計測前に検査する。
- 各ケースで最初の実行を別保存し、さらに全64tokenの暖機を1回行い、3回計測する。
  各回のKVは新規。最初のケースだけがプロセス内初回で、driver/OS cacheの消去はしていない。
  主実験はモデルごとにDeno→PyTorch f32→BF16の固定順。QAT textは追加走行で、交互順序のA/Bではない。
- TTFT = リクエスト開始から最初の非停止token IDがホストへ届くまで。
  decode tok/s = `(配送数 − 1) / (完了 − 最初の配送)`。停止tokenを配送数に含めない。
  モデル読込・tokenize・文字列復号・端末表示・外側のsequence解放は除外し、sequence/cache作成は含める。
  Denoのstream完了前に内部で行うcontext解放は含まれ、PyTorchのcache参照解放は計測後。
  各回の直前にGPUの完了を待ち、前回の仕事を混ぜない。
- PyTorchはSDPA attention、dynamic KV cache、`torch.compile`なし、CPU 4 thread。
  float32 matmul precisionは`highest`。Denoのchunk長は64。
- 通常版は保存済みINT4/INT8の整数値とscaleをPyTorchへ復元するが、GPUではdense f32/BF16へ展開する。
  Denoのpacked整数格納とは異なる。BF16は追加の丸めを伴う別条件として表示する。
- 通常GemmaのPLEはPyTorchではCPU行取得とGPUへの行転送、Denoでは既定の2shard分のホストcache。
  QATのPyTorchはpacked PLEをGPUに保持する。配置差を速度だけから消去したことにはしない。

## QAT の公式経路の切り分け

最初は公式`Gemma4ForConditionalGeneration`をそのまま使用した。
E2Bは生成できたが、E4Bは初回の実行でCUDAメモリ不足になった。
スタックは`Gemma4Model.forward`のPAD埋め込み取得から`QuantizedEmbedding.weight`へ入り、
1行を選ぶ前に表全体の量子化を解除する箇所を示した。E4Bでは追加2.50 GiBの確保に失敗した。
これは保存重みがGPUに載らない通常E4B f32の事前拒否とは別である。

テキストだけの比較として、公式`Gemma4ForCausalLM`のモデルに、検証済みQAT checkpointの
`language_model`と`lm_head`を同じオブジェクトのまま接続する条件を追加した。
丸め・INT2/INT4/INT8の量子化層・PLE・logitsのsoftcapは公式実装のままである。
不要なvision/audioの重みとmultimodal wrapperを外し、PAD行のための表全体展開を避ける。
ツールではQATの既定をこの`text`条件とし、元経路は`--qat-model conditional`で明示できる。
両条件を同じ結果として混ぜない。外部の実装ソースは複製していない。

QATの接続変更はCPU float32/eagerで、E2B/E4B × 英語/日本語の4条件を確認した。
各条件のprefill最終logitsと、KVを引き継ぐ次の1tokenの全262,144語彙logitsが`torch.equal`で一致した。
重みのParameterは同一オブジェクトで、元の量子化層を浮動小数点層へ交換していない。
検証は`outputs/bench/karume/2026-09-12_qat-text-speed-r37k5w/check.py`と`result.json`に保存した。

## 暖機後の実測

各値は3回の中央値。セル内は**英語 / 日本語**。全て64tokenを配送し、反復内の出力列は一致した。
通常E4BのPyTorch列だけはBF16で、それ以外のPyTorch列はfloat32。QATは公式テキスト経路。

| モデル         |    Deno tok/s | PyTorch tok/s | Deno TTFT ms | PyTorch TTFT ms |
| -------------- | ------------: | ------------: | -----------: | --------------: |
| qwen3-06b      | 45.43 / 44.61 | 50.73 / 51.27 |  90.5 / 92.1 |     22.1 / 22.3 |
| minicpm5-2b    | 36.77 / 36.43 | 40.96 / 40.98 | 103.9 / 88.5 |     28.3 / 28.4 |
| gemma4-e2b     | 38.00 / 37.93 | 32.20 / 31.90 |  50.9 / 71.4 |     38.2 / 41.4 |
| gemma4-e4b     | 27.57 / 27.33 | 23.83 / 23.72 | 83.5 / 129.9 |     46.2 / 48.7 |
| gemma4-qat-e2b | 37.55 / 37.26 |   8.59 / 8.28 |  52.9 / 73.7 |   122.4 / 132.9 |
| gemma4-qat-e4b | 27.13 / 27.29 |   4.43 / 4.96 | 89.8 / 123.0 |   263.5 / 255.0 |

通常モデルのPyTorch BF16補助比較。QATは固定SRQを保つためBF16へ変更していない。

| モデル      | tok/s（英 / 日） | TTFT ms（英 / 日） |
| ----------- | ---------------: | -----------------: |
| qwen3-06b   |    55.06 / 53.30 |        19.0 / 19.4 |
| minicpm5-2b |    46.77 / 45.92 |        21.9 / 22.5 |
| gemma4-e2b  |    30.43 / 30.70 |        37.3 / 38.4 |
| gemma4-e4b  |    23.83 / 23.72 |        46.2 / 48.7 |

最初の英語ケースのTTFT。プロセス内初回の1点で、完全なcold測定ではない。

| モデル         | Deno ms | PyTorch ms |
| -------------- | ------: | ---------: |
| qwen3-06b      |   158.2 |      400.3 |
| minicpm5-2b    |   351.3 |      461.5 |
| gemma4-e2b     |   184.0 |      410.4 |
| gemma4-e4b     |   280.4 |      554.6 |
| gemma4-qat-e2b |   257.3 |      683.6 |
| gemma4-qat-e4b |   339.8 |      758.0 |

QAT E2Bの元のconditional経路は 7.53 / 7.54 tok/s、TTFT 146.7 / 159.6 ms。
追加したtext経路とのCUDAの64token列は英語・日本語とも一致した。通常E4Bのf32は重みだけで
18,577,764,352 B必要で、空き12,216,762,368 Bを超えるため実行前に拒否した。
QAT E4B conditionalの初回OOMは別の失敗として保存し、成功値へ置き換えていない。

## 出力一致と解釈

全16成功構成 × 2ケース × 初回/暖機/3反復の**160生成**を保存し、いずれも64tokenだった。
各構成の反復は全列一致した。DenoとPyTorchの比較は次のとおり。

- Qwen3はf32/BF16とも英語・日本語の64tokenが一致。
- MiniCPM5と通常Gemma E2Bのf32は両言語で一致。BF16の日本語はそれぞれ先頭18/37tokenまで一致し、その次で分岐。
- 通常Gemma E4BのBF16は両言語で一致。f32は容量不足のため比較していない。
- QAT E2Bは両PyTorch経路とも英語22token、日本語56tokenまで一致し、その次でDenoと分岐。
  QAT E4Bのtext経路は英語64tokenが一致、日本語は43tokenの次で分岐。
  QAT E2BのPyTorch conditional/text間ではCUDAの全生成列が一致した。

今回の分岐箇所について誤差の演算別帰属までは行っていない。
QATの縮約差がSRQの丸め境界をまたぐ現象は既に別の短文で検証しているが、
この入力でも同じ原因と断定しない。既存の許容誤差・golden・期待列は変更していない。
速度比較では配送数と系列長を揃えたが、出力列の違いによるPLEアクセスの差などは残る。

実測として、Qwen/MiniCPMにはPyTorchとの差があり、特にTTFTの差がdecodeの差より大きい。
通常Gemmaではこの条件のDeno decodeがPyTorch f32/BF16より速く、QATでもDenoが速い。
通常E2BではPyTorchのBF16化がf32より速くならなかった。この比較だけでは律速箇所は確定しない。

コード上の事実として、MiniCPM/QwenのDeno実験グラフは64行のprefillで全行のlogitsとtokenを出し、
通常の`session.run`が全出力をホストへ戻す。一方PyTorchは`logits_to_keep=1`で最終行だけを投影する。
**推測**: 不要な出力投影・転送・固定chunkの計算がTTFT差の一部である可能性が高い。
次は投影・GPU実行・readback・ホスト処理の各時間を採って帰属し、必要な出力を減らす候補を比べる。
基準値の作成中にruntimeや保存グラフは変更していない。

M2ではこのツールのDenoと`--device mps`を同じ入力・保存重みで測る必要がある。
MPSのコードは用意したが、ここで実行確認したのはCUDAだけである。
今回のRTX値やこのPyTorch実装の値を、WebML/ChromeやApple GPUの上限とは扱わない。

## 生データと検証

- 主実験: `outputs/bench/karume/2026-09-12_llm-speed-final-WAkJev/`。
  `run-matrix.py`と`matrix.log`、各構成の全run JSON、ソースの凍結コピーを保存。
  最後のQAT E4B conditionalで失敗したため、主controllerは成功扱いで閉じていない。
- QAT textの追加: `outputs/bench/karume/2026-09-12_qat-text-speed-r37k5w/`。
  CPUの接続検証とGPUの2構成を保存。`matrix.json`が追加走行の完了記録。
- 再集計: 主実験の`aggregate-v2.py` / `aggregate-v2.json`。
  個別runとsummaryを照合し、時間・配送数から速度を再計算して一致を検査した。
  元の`aggregate.py`は主実験成功を前提とする未実行版として残し、上書きしていない。
- 重み・config・tokenizer・PLEの93ファイルを保存SHAと再照合した。
  `check-weight-hashes.py` / `verified-weight-hashes.json`が正本。
  依存の全versionは`python-freeze.txt`、PyTorch環境は別ディレクトリのvenvに隔離した。
- 追跡する集計は[llm-speed-results.json](2026-09-12-llm-speed-results.json)。
  全反復のmin/median/max、初回、配置、出力一致、元summaryと集計JSONのSHAを含む。
- 試走と準備失敗は`outputs/bench/karume/2026-09-12_llm-speed-5hw2Ht/`に保存。
  この節の表は試走値を混ぜていない。公式会話テンプレートはモデル資産として実験先に置き、gitへ複製していない。

速度計測の時計4件と既存の品質復元/採点18件は**22 passed / 0 failed**。
QAT追加後のログは追加実験の`python-tests.log`。Ruffのlintとformat検査も成功した。

全体検証は**2,934 passed（760 steps）/ 0 failed / 5 ignored、25分20秒**。
`outputs/bench/karume/2026-09-12_llm-speed-verify-fNWKU5/verify.log`に保存した。
GPUベンチは終了後に検証を開始し、並走していない。
保存集計と全16summaryのSHA・相対リンクを機械照合した（同ディレクトリの`check-record.log`）。
Deno runner・入力準備・最終PyTorch runnerは計測時の凍結コピーと一致し、
`measured-source-hashes.json`に最終ソースのSHAを残した。
