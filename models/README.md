# models/ — ローカル資産の索引（この README 以外は git 対象外）

2 層に分かれる。**配布形**（`anima-turbo/` — そのまま HF へ上げられる 1 リポ）と、その**源になる
系列ディレクトリ**（エクスポータが吐いた生の出力）。デモや計測の**生成物は `models/` に置かず
リポ直下の `outputs/`** へ出す（資産と出力を同じ木に混ぜない）。

## 配布形 — `anima-turbo/`

`karume.json` + ADR 0038 §2 の規約名で並んだファイル群。ここに入るのは manifest が宣言した
ファイルだけで、系列に並ぶ E2E フィクスチャ（`io.*.safetensors`）は**入らない**。

```
cd tools/exporter && uv run karume dist
```

組み立ては冪等（再実行で貼り直す）。同一ファイルシステムなので実体は**ハードリンク**で、
系列と配布形が二重にディスクを食うことはない。

| パス                                 | 出所                                                        |
| ------------------------------------ | ----------------------------------------------------------- |
| `karume.json`                        | 組み立て時に**実ファイルから導出**（手書き禁止 — ADR 0038） |
| `text_encoder/model.safetensors`     | `anima-f16/text_encoder/`                                   |
| `text_conditioner/model.safetensors` | `anima-f16/text_conditioner/`                               |
| `transformer/model.f16.safetensors`  | `anima-turbo-f16-dyn/transformer/`                          |
| `transformer/model.i8.safetensors`   | `anima-turbo-i8-dyn/transformer/`                           |
| `transformer/rope_base.safetensors`  | 上記 2 系列（**バイト同一を検証して 1 本化**）              |
| `vae_decoder/model.safetensors`      | `anima-f16/vae_decoder/`                                    |
| `tokenizer/qwen2-tokenizer.json`     | `anima-demo/text/`                                          |
| `tokenizer_2/t5-tokenizer.json`      | `anima-demo/text/`                                          |

`rope_base` が系列間で食い違ったら組み立ては**止まる**（片方を黙って選ぶと、選ばれなかった
系列の preset が別の幾何の rope 表で走り、ロードも実行も通って絵だけが壊れる）。

## 系列ディレクトリ（組み立ての源 — 消さない）

再生成コマンドの正本は [tools/exporter/README.md](../tools/exporter/README.md)。E2E は資産が
無い系列を **SKIP** する（部分欠落は FAIL）ので、削除した系列のテストは再 emit まで走らない。

| ディレクトリ           | 内容                                                          |
| ---------------------- | ------------------------------------------------------------- |
| `anima-f16/`           | text_encoder / text_conditioner / VAE decoder（f16）+ io 参照 |
| `anima-turbo-f16-dyn/` | turbo DiT f16 **S 形**（解像度非依存）+ rope 素表 + io 参照   |
| `anima-turbo-i8-dyn/`  | turbo DiT i8 **S 形** + rope 素表 + io 参照                   |
| `anima-demo/text/`     | トークナイザ表（Qwen2 BPE / T5 Unigram）                      |

配布するのは S 形 1 本だけ（固定形は配布しない — ADR 0038 §4）。

## `outputs/`（リポ直下・git 対象外）

デモと計測の生成物の置き場。PNG・WAV・dump・目視ゲートの記録画像はここへ出す。`models/` は
「入力（資産）」だけを持ち、`outputs/` が「出力」を持つ — 片方を丸ごと消しても、もう片方の
再生成手順が変わらない分け方にしてある。
