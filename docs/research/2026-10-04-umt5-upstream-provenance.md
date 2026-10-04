# umT5 配布リポの上流を google/umt5-xxl にできるか — 出所の実測

> 時点スナップショット（2026-10-04）— karume-umt5-xxl の上流を Wan 同梱の umT5 から本家 google/umt5-xxl へ移せるかを、重み・トークナイザ・容器への影響まで実測で確定した調査記録（recipe と配布物は未変更）。

使った版: transformers 5.14.1・torch 2.13.0+cpu・huggingface_hub 1.27.0・hf-xet 1.6.0（`tools` の venv・`uv run --group wan --inexact`）。
CPU のみ。作業ファイルは `/tmp/claude-1000/recon-2026-10-04/umt5-upstream/`（リポ外）。

## 0. 結論

- **export できる。しかも結果は今と同じ値になる。** google/umt5-xxl（commit `66cb9e7e…`）の encoder 243 本は、Wan2.1 Diffusers の
  `text_encoder` 242 本と **f32 で全要素がビット一致**した（差のある要素 0 / 6,731,059,200。google 側だけの `encoder.embed_tokens.weight` は
  `shared.weight` と同じ storage）。Wan の umT5 は google の encoder をそのまま写したもの（f32・無加工）と言える。
- 量子化の入力が同一なので、i8 の packed と scale も同一（recipe の `quantize_rows` を両方から呼んで 3 本で実測一致）。trace したグラフも
  google の config から組んで同一。**容器で変わるのは part 0 の `provenance.upstreamRevision`（40 桁の revision）だけ**の見込み（推測 — 容器は書いていない）。
- 障害は 3 つ（どれも recipe 側の口の問題）: ① google は **pickle 形式の `.bin` だけ**（safetensors は未マージの bot PR にしか無い）、
  ② google の索引は `shared.weight` と `encoder.embed_tokens.weight` の両方を持ち、今の `checkpoint_keys` が「1 つに決まらない」で落ちる（実測）、
  ③ `UMT5EncoderModel.from_pretrained` は decoder を含む 6 本全部（約 51.9 GB）を要求する（実測で 4 本目が無くて落ちる）。
- トークナイザは google 版と Wan 版で id 列が同じ（2,034 本で不一致 0）。ただし `tokenizer.json` のスコアが 65,654 個ずれる（最大 3.55e-15）
  ので、Karume のトークナイザ資産を google から焼くとバイトが変わる。資産は Wan のリポに置くものなので、出所は Wan のままがよい（§5）。
- Wan2.2 TI2V-5B（Diffusers）の umT5 は **bf16 で格納**され、Wan2.1 の f32 を最近接偶数丸め（RNE）で bf16 にした値と全 242 本でビット一致。
  Wan 公式（非 Diffusers）の 2 リポは同じ `models_t5_umt5-xxl-enc-bf16.pth`（LFS sha256 一致）を持つ。トークナイザも同一。
  **Wan2.2 TI2V-5B は karume-umt5-xxl の容器をそのまま越境参照できる**。

## 1. google/umt5-xxl の HF 上の実体（事実）

一次ソース: <https://huggingface.co/api/models/google/umt5-xxl?blobs=true>（HTTP 200）と、同 revision の `resolve/` で取った各 JSON。

| 項目           | 値                                                                                                               |
| -------------- | ---------------------------------------------------------------------------------------------------------------- |
| commit SHA     | `66cb9e7e85526fe440a945569e42c72fb6cbc0ad`（main。tags なし — `/api/models/google/umt5-xxl/refs`）               |
| lastModified   | 2023-07-03T05:37:17Z                                                                                             |
| ライセンス     | `apache-2.0`（API の `license` と README front matter の `license:` の両方）。LICENSE / NOTICE ファイルは無い    |
| 形式           | `pytorch_model-0000N-of-00006.bin`（**pickle・zipfile 形式** — `zipfile.is_zipfile` が真）6 本。safetensors 無し |
| 索引           | `pytorch_model.bin.index.json`: `total_size` 51,890,126,848 B・キー 606 本                                       |
| アーキテクチャ | `UMT5ForConditionalGeneration`（encoder + decoder + `lm_head`）                                                  |

shard ごとのサイズと LFS sha256（API の値。取得した 1〜3 本は手元の `sha256sum` と一致）:

| shard | サイズ (B)    | LFS sha256（先頭 16 桁） | encoder + `shared` のキー | decoder + `lm_head` のキー  |
| ----- | ------------- | ------------------------ | ------------------------- | --------------------------- |
| 1     | 9,871,633,465 | `382094214dfe74d7`       | 78                        | 1（`decoder.embed_tokens`） |
| 2     | 9,966,219,296 | `b49efce006c907ea`       | 127                       | 0                           |
| 3     | 9,999,835,818 | `da3d39fffe646424`       | 38                        | 103                         |
| 4     | 9,999,829,533 | `9da344dda8103ca0`       | 0                         | 146                         |
| 5     | 7,852,227,059 | `0ad06915eba0878c`       | 0                         | 112                         |
| 6     | 4,200,596,394 | `5798b8aa388b4d3e`       | 0                         | 1（`lm_head`）              |

- encoder に要るのは shard 1〜3（計 29,837,688,579 B）。この 3 本だけを HF の既定キャッシュへ落とした。
- 索引には `shared.weight`・`encoder.embed_tokens.weight`・`decoder.embed_tokens.weight`・`lm_head.weight` が別名で載る
  （`jq '.weight_map|keys[]'`）。encoder 側の 243 本 = `shared.weight` + `encoder.*` 242 本（`encoder.embed_tokens.weight` を含む）。
- safetensors 版は **未マージの PR だけ**: `refs/pr/2`（`ae98d537…`）と `refs/pr/3`（`538e8207…`）。どちらも作者 `SFconvertbot`・
  2024-11-14 作成・open（`/api/models/google/umt5-xxl/discussions`）。shard 1 の sha256 が 2 つの PR で違う（`c2f28c28…` / `0ef7767d…` —
  2〜6 本目は同じ）。理由は未調査。

### 1.1 config.json の違い

google の `config.json` と Wan2.1 Diffusers の `text_encoder/config.json` の差（`jq -S` の目視と `UMT5Config.to_dict()` の差分）:

| 欄                     | google/umt5-xxl                | Wan2.1 Diffusers `text_encoder` |
| ---------------------- | ------------------------------ | ------------------------------- |
| `_name_or_path`        | `"."`                          | `"google/umt5-xxl"`             |
| `architectures`        | `UMT5ForConditionalGeneration` | `UMT5EncoderModel`              |
| `model_type`           | **欠けている**                 | `"umt5"`                        |
| `classifier_dropout`   | 欠けている                     | 0.0                             |
| `transformers_version` | 4.31.0.dev0                    | 4.48.0.dev0                     |

形の欄（層数 24・`d_model` 4,096・`d_ff` 10,240・heads 64・`d_kv` 64・buckets 32・max_distance 128・`vocab_size` 256,384・`gelu_new`・
`torch_dtype` float32 など）は同じ。`UMT5Config.to_dict()` を比べると差は `_name_or_path` と `architectures` だけだった
（transformers 5.14.1 は両方の `tie_word_embeddings` を True と読む — ファイルは両方 false）。

実測（`cfg_check.py`）:

- `UMT5Config.from_pretrained(<google の snapshot>)` は通る（recipe の `meta_text_encoder` が使う口 — `umt5_export.py:357-364`）。
- `AutoConfig.from_pretrained(<google>)` は落ちる: `ValueError: Unrecognized model in … Should have a model_type key in its config.json.`
- `UMT5EncoderModel.from_pretrained(<google>, dtype=bf16)` は shard 1〜3 だけでは落ちる:
  `FileNotFoundError: … pytorch_model-00004-of-00006.bin`（索引に載る 6 本全部を開きに行く）。

## 2. 重みの同一性（事実・実測）

### 2.1 google の encoder と Wan2.1 Diffusers の umT5

`weights_compare.py`: google の shard を `torch.load(mmap=True, weights_only=True)` で 1 本ずつ開き、Wan 側は `safe_open` で 1 テンソルずつ読んで
突き合わせた（f32 のビット列の比較・bf16〈RNE〉に落とした後のビット列の比較・f64 での最大絶対差と相対 RMS。2^24 要素ずつの塊で計算）。

| 項目                                           | 値                                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------------- |
| google の encoder 側のキー                     | 243（`shared.weight` + `encoder.*` 242）                                        |
| Wan のキー                                     | 242（`shared.weight` + `encoder.*` 241）                                        |
| google だけにあるキー                          | `encoder.embed_tokens.weight`（→ Wan の `shared.weight` と照合）                |
| Wan だけにあるキー                             | なし                                                                            |
| 形・dtype                                      | 243 本すべて同じ形・両方 float32                                                |
| f32 でビット一致したテンソル                   | **243 / 243**（差のある要素の総数 0 / 6,731,059,200 — 埋め込みを 2 回数えた数） |
| bf16 に落とした後のビット一致                  | 243 / 243                                                                       |
| 最大絶対差・最大相対 RMS                       | 0.0・0                                                                          |
| google 内の `shared` と `encoder.embed_tokens` | 同じ storage（`untyped_storage().data_ptr()` が一致）・値も一致                 |
| 所要                                           | 150.5 s                                                                         |

相対位置バイアスの表 24 本（`[32,64]`）・RMSNorm の重み 49 本・語彙埋め込み `[256384,4096]` を含めて一致した。

### 2.2 Wan2.1 Diffusers の f32 は bf16 からの昇格ではない

`wan21_bf16_check.py`: Wan2.1 Diffusers の全 242 本で、f32 の下位 16 ビットが 0 でない要素は 5,680,823,446 / 5,680,910,336。
242 本すべてに非 0 の要素がある。bf16 の値を f32 に広げたものではなく、本物の f32（§2.1 から google の f32 そのもの）。
ADR 0119 追記 D（`docs/decisions/0119-wan-umt5-gpu-text-encoder.md:486-487`）の「bf16 の写しを F32 で格納したもの」は
同 ADR の `:516` で既に訂正済みで、今回の実測もそれと合う。

## 3. トークナイザの同一性（事実・実測）

### 3.1 ファイルの sha256

| ファイル                  | google/umt5-xxl `66cb9e7e`  | Wan2.1 Diffusers `0fad780a` `tokenizer/` | Wan 公式 Wan2.1-T2V-1.3B `37ec5126` `google/umt5-xxl/` |
| ------------------------- | --------------------------- | ---------------------------------------- | ------------------------------------------------------ |
| `spiece.model`            | `e3909a67…`（4,548,313 B）  | 同じ                                     | 同じ（API の LFS 値）                                  |
| `tokenizer.json`          | `af904105…`（16,853,013 B） | `20a46ac2…`（16,837,459 B）              | `6e197b4d…`（16,837,417 B）                            |
| `tokenizer_config.json`   | `616015e2…`                 | `1d8d2a21…`                              | `ed9a3a8b…`                                            |
| `special_tokens_map.json` | `14ef492a…`                 | `456b58fd…`                              | `7b8a9f50…`                                            |

Wan2.2 TI2V-5B Diffusers（`b8fff731`）の `tokenizer/` は `spiece.model` `e3909a67…`・`tokenizer.json` `20a46ac2…`（Wan2.1 Diffusers と同じ）で、
`tokenizer_config.json` も Wan2.1 Diffusers と `cmp` で同一。Wan 公式 Wan2.2-TI2V-5B（`921dbaf3`）の `google/umt5-xxl/` は
Wan 公式 Wan2.1 と同じ sha256（API の LFS 値）。

### 3.2 中身の比較（`tok_compare.py`）

recipe の `umt5_tokenizer.load_truth` / `check_upstream_shape` をそのまま使い、3 系統を Wan2.1 Diffusers と比べた
（入力 = 固定 4 プロンプト〈正規化後〉 + `BOUNDARY_CASES` 30 本 + 乱択 2,000 本 = 2,034 本）。

| 項目                                          | google                                     | Wan 公式       |
| --------------------------------------------- | ------------------------------------------ | -------------- |
| `check_upstream_shape`（TS が前提にする形）   | 通る                                       | 通る           |
| 語彙 256,300 ピースの並び                     | 同一                                       | 同一           |
| スコア                                        | **65,654 個が違う・最大差 3.55e-15**       | 完全一致       |
| 追加語彙・pre_tokenizer（組み直し後）・unk_id | 同一（unk_id 2）                           | 同一           |
| `tokenizer.json` の生の pre_tokenizer         | 旧書式 `Metaspace{add_prefix_space: true}` | 旧書式（同左） |
| id 列の不一致（transformers 5.14.1 の経路）   | 0 / 2,034                                  | 0 / 2,034      |
| id 列の不一致（`tokenizer.json` の経路）      | 0 / 2,034                                  | 0 / 2,034      |
| 固定 4 本の長さ                               | 28 / 118 / 50 / 126                        | 同左           |

Wan2.1 Diffusers の生の pre_tokenizer は新書式（`prepend_scheme: always`・`split: true`）。正規化（`Replace(" {2,}" → " ")`）は 3 系統で同じ。

### 3.3 Karume のトークナイザ資産との関係

- 資産 `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json`（8,117,546 B・sha256 `b5fb3658…`）と配布形
  `models/karume-wan2.1/t2v-1.3b/umt5_tokenizer/tokenizer.json` はバイト同一。資産の `source` 欄は
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`・`0fad780a…`・`tokenizer`（`jq`）。書き手は `umt5_tokenizer.build_asset`（`umt5_tokenizer.py:280-308`）で、
  入力は `tokenizer_snapshot`（`:144-166` — Wan の pin）。
- karume-umt5-xxl はトークナイザを持たない（`umt5_distribution.py:93-94` の `UMT5_ASSETS = {}`）。資産は Wan のリポの部品。
- google の `tokenizer.json` から焼くと `scores` 欄（`build_asset` の `float(score)`）が 65,654 個変わるので、資産のバイトが変わる
  （`asset_scores_equal: false` — 実測）。`source` 欄も変わる。id 列は上の範囲で同じ。

## 4. 本家を上流にしたとき配布物が変わるか

### 4.1 事実（実測）

- `quant_compare.py`: recipe の `quantize_rows`（`umt5_export.py:401-429`）を、Wan 側は `Checkpoint`、google 側は同じ口
  （`shape` / `read_rows` / `names`）を持つ mmap の .bin 読み手から呼び、packed（i8）と scale（f32 のビット列）を比べた。

  | テンソル                                            | 形              | packed | scale |
  | --------------------------------------------------- | --------------- | ------ | ----- |
  | `encoder.block.0.layer.0.SelfAttention.q.weight`    | `[4096,4096]`   | 一致   | 一致  |
  | `encoder.block.23.layer.1.DenseReluDense.wo.weight` | `[4096,10240]`  | 一致   | 一致  |
  | `shared.weight`（語彙埋め込み）                     | `[256384,4096]` | 一致   | 一致  |

- `trace_compare.py`: Wan の config と google の config から組んだ meta の `UMT5EncoderModel` を `umt5_export.trace` に通すと、
  `IrGraph`（dataclass の `==`）・テンソル表（名前・形・dtype）・量子化の対象（169 本・軸）がすべて一致（ノード 914・テンソル 244）。
- 今の容器の出所（`_shared.container_read.read_provenance`）: `license='apache-2.0'`・`notice='NOTICE.md'`・
  `upstream_revision='0fad780a534b6463e45facd96134c9f345acfa5b'`。container-v1 §2.3（`docs/container-v1.md:259-266`）の `provenance` は
  revision だけを持ち、**リポ名は持たない**。`provenance` はモデル記述（part 0）の中にある（`docs/glossary.md:49`）。
- golden（`reference.*.safetensors`）のメタは part 0 の sha256 を持ち（`umt5_reference.py:673`・`:795`）、TS がそれを今の容器と突き合わせる
  （`packages/models/tests/e2e_wan_umt5_test.ts:620`）。
- Wan の配布形 `models/karume-wan2.1/karume.json` は umT5 の容器を part ごとの sha256・サイズとモデル記述の sha256 で越境参照する
  （`jq` で確認 — `repo: hdae/karume-umt5-xxl`・revision は開発用の 0 埋め）。

### 4.2 推測（容器は書いていない）

- 全 243 本の f32 が同一で、量子化はその決定的な関数なので、`fixed_weights` と量子化しない重みはすべて同じ値になる。グラフも同一。
  よって **part 2〜26（重みの block）はバイト同一**、**part 0 は `upstreamRevision` の 40 文字だけが変わる**見込み。revision は同じ 40 桁なので
  モデル記述の長さも同じはず。
- 焼き直しの範囲:

  | 対象                                                        | 変わるか                              | 理由                            |
  | ----------------------------------------------------------- | ------------------------------------- | ------------------------------- |
  | 系列の容器 part 0（`model-00001-of-00026.krm`）             | 変わる                                | `provenance.upstreamRevision`   |
  | 系列の容器 part 2〜26                                       | 変わらない                            | 重み・グラフが同一              |
  | 系列の golden（今は `reference.accept-*.safetensors` 4 本） | メタの `part0Sha256` だけ変わる       | 参照の値は重みが同じなので同じ  |
  | `models/karume-umt5-xxl/karume.json`                        | part 1 の sha256・モデル記述の sha256 | part 0 の変化                   |
  | `models/karume-wan2.1/karume.json`（越境参照）              | 同上 + 参照先 revision（公開後）      | 同上                            |
  | karume-umt5-xxl の README / NOTICE                          | 変わる                                | 出所の文面（§5）                |
  | GPU 経路の sha 参照行（ADR 0106）・Wan の DiT / VAE / 資産  | 変わらない                            | 計算の入力（重み・id 列）が同じ |

  `write --check`（`umt5_export.py:567-593`）は part 0 の不一致で落ちるはずなので、移行の回は `write` → `reference` → `dist` の順で書き直し、
  part 2〜26 が不変であることを差分で確かめるのが検証になる。

## 5. recipe の変更点

### 5.1 事実（今のコードが Wan の pin に縛っている箇所）

| 箇所                                                                                      | 今の形                                                                                                     | google を上流にしたときの問題                                                                                                                             |
| ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wan/sources.py:54-60` `SOURCES`                                                          | Wan のモデル名 → Wan のリポ・revision・license の表 1 本                                                   | umT5 用の行が無い（表のキーは Wan の配布形のモデル名）                                                                                                    |
| `wan/sources.py:140-169` `text_snapshot`                                                  | Wan のリポの `text_encoder/*`・`tokenizer/*` を取る                                                        | google はリポ直下に置く（subfolder が無い）・取るべきは shard 1〜3 + 索引 + config                                                                        |
| `wan/umt5_export.py:101` `UPSTREAM_SUBFOLDER = "text_encoder"`・`:352-354` `upstream_dir` | snapshot の下の `text_encoder`                                                                             | google ではリポ直下                                                                                                                                       |
| `wan/umt5_export.py:259-307` `Checkpoint`                                                 | `model.safetensors.index.json` か `model.safetensors` だけを受ける（無ければ fail loudly）                 | google は `pytorch_model.bin.index.json` + `.bin`                                                                                                         |
| `wan/umt5_export.py:309-329` `checkpoint_keys`                                            | tied な別名のうち checkpoint に在るものが 1 つでないと fail loudly                                         | google は 2 つとも在る → 実測で `'encoder.embed_tokens.weight' の checkpoint のキーが 1 つに決まらない: ['encoder.embed_tokens.weight', 'shared.weight']` |
| `wan/umt5_export.py:512-517` `provenance`                                                 | `SOURCES[model]` の license と revision を焼く                                                             | Wan の revision を焼く                                                                                                                                    |
| `wan/umt5_export.py:618-633` `load_upstream`（`check-mask` 用）                           | `UMT5EncoderModel.from_pretrained(snapshot, subfolder="text_encoder")`                                     | google では 6 本全部（約 51.9 GB）が要る（§1.1 の実測）                                                                                                   |
| `wan/umt5_export.py:730-758` `reference_summary`                                          | ケースの id 列は `umt5_reference.upstream_encoder`（Wan の tokenizer）・`checkpoint_weights(upstream_dir)` | 後者は `checkpoint_keys` を通るので同じ別名の問題                                                                                                         |
| `wan/umt5_distribution.py:154-158` `UMT5_UPSTREAM`                                        | `Umt5Upstream(SOURCES[DEFAULT_MODEL], "text_encoder")`                                                     | 出所の表の行を差し替える場所                                                                                                                              |
| `wan/umt5_distribution.py:285-288` `assert_upstream_provenance`                           | 容器の provenance を上の表の license / revision と突き合わせる                                             | 表を変えれば容器も焼き直しが要る（門が落とす — 意図どおり）                                                                                               |
| `wan/umt5_distribution.py:331-337` NOTICE・`:396-407` frontmatter・`:434-454` 帰属節      | 「Wan2.1 の `text_encoder`」・`base_model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers`・「同一は未確認」             | 出所の文面                                                                                                                                                |
| `wan/umt5_distribution.py:1-11` モジュール doc・`wan/README.md:370`・ADR 0119 `:485-487`  | 「google との同一は確かめていない」                                                                        | 今回の実測で事実が変わった                                                                                                                                |
| `wan/tests/test_umt5_distribution.py:33`・`:60-62`・`:245-246`・`:273`                    | `SOURCES[DEFAULT_MODEL]` を出所の期待値に使う                                                              | 期待値の参照先                                                                                                                                            |
| `wan/card.py:228-240`（Wan のカード）                                                     | 「the umT5-XXL encoder of the same checkpoint」                                                            | 出所が google になると文面がずれる（値は同一なので嘘ではないが、出所の表と食い違う）                                                                      |
| 系列名 `wan2.1-umt5-i8-dyn`（`umt5_export.py:89`・`umt5_distribution.py:73`）             | Wan を名乗る                                                                                               | 必須ではない（下の判断）                                                                                                                                  |

### 5.2 推奨の形（推測を含む設計案）

- **上流の表**: `sources.py` に umT5 専用の表（例 `UMT5_SOURCES = {"xxl": UpstreamSource("google/umt5-xxl", "66cb9e7e85526fe440a945569e42c72fb6cbc0ad", "apache-2.0")}`）
  と、encoder の shard だけを取る取得口を足す。取るファイルは索引の `weight_map` から encoder と `shared` のキーが載る shard を導く
  （今は 1〜3 — 綴りを手で書かない）+ `config.json` + `pytorch_model.bin.index.json`。`UMT5_UPSTREAM` の `subfolder` は空（リポ直下）にする。
- **読み口**: `Checkpoint` に `.bin` の分割形を足す（`pytorch_model.bin.index.json` → `torch.load(path, map_location="cpu", mmap=True, weights_only=True)`。
  shard ごとに 1 回開いて使い回す）。今回の比較で 1 本ずつの読み出しと行の切り出しが RSS を抑えて動いた（243 本で 150.5 s）。
  F32 の検査は `torch.float32` の文字列で既に通る（`umt5_export.py:303-306`）。
- **別名**: 候補が 2 つ以上のとき、全候補が同じ storage（またはビット一致）なら transformers の tied の宣言
  （`UMT5EncoderModel._tied_weights_keys = {'encoder.embed_tokens.weight': 'shared.weight'}` — 実測）の先 `shared.weight` を選び、違えば fail loudly。
  今の「黙って別の重みを読まない」という意図（`umt5_export.py:316-317`）は保てる。
- **check-mask**: 調べたいのは Wan の事前計算資産の経路なので、Wan の snapshot のまま残すのが自然（google を読むと 6 本全部が要る）。
- **トークナイザ**: Wan のリポの資産で、Wan の前処理の表と束ねた形式（ADR 0119 追記 C）。出所は Wan のままにする。id 列は同じだが、
  google から焼くとスコアの桁でバイトが変わり、資産の焼き直しと TS の fixture の確認が増えるだけで得るものが無い。
- **カード**: `base_model: google/umt5-xxl`・帰属は「the encoder of google/umt5-xxl at commit `66cb9e7e…`（decoder と `lm_head` は使わない）」。
  「Relation」の行は「Wan2.1 Diffusers の `text_encoder`（f32）と全テンソルがビット一致・Wan2.2 TI2V-5B Diffusers の bf16 はその RNE 丸め
  （2026-10-04 に確認）」へ。NOTICE の冒頭も同じ書き換え。google のリポには NOTICE ファイルが無いので、引き継ぐ NOTICE の本文は無い
  （LICENSE.md は今と同じ Apache 2.0 の原文）。

## 6. Wan2.2 TI2V-5B 同梱の umT5（事実・実測）

一次ソース: `/api/models/<repo>?blobs=true`（4 リポとも HTTP 200）。

| リポ（commit）                                 | umT5 のファイル                                                                | 格納                              | ライセンス |
| ---------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------- | ---------- |
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers（`0fad780a`） | `text_encoder/` safetensors 5 本・`total_size` 22,723,641,344 B                | F32                               | apache-2.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers（`b8fff731`）  | `text_encoder/` safetensors 3 本・`total_size` 11,361,820,672 B                | BF16（77 / 127 / 38 本 — ヘッダ） | apache-2.0 |
| Wan-AI/Wan2.1-T2V-1.3B（`37ec5126`・公式）     | `models_t5_umt5-xxl-enc-bf16.pth` 11,361,920,418 B・sha256 `7cace0da2b446bbb…` | bf16（名前から）                  | apache-2.0 |
| Wan-AI/Wan2.2-TI2V-5B（`921dbaf3`・公式）      | 同名・同サイズ・**同じ sha256**                                                | 同左                              | apache-2.0 |

- Wan2.2 Diffusers の 3 本の sha256 は手元で API の LFS 値と一致（`a8e86196…` / `d57d948e…` / `0da9ee28…`）。キー 242 本は Wan2.1 Diffusers と同一。
  `config.json` の差は `torch_dtype`（bfloat16 / float32）だけ。
- LFS の sha256 は格納が違う（F32 と BF16）ので一致しない。値は `w22_compare.py` で比べた: Wan2.2 の bf16 は Wan2.1 の f32 の
  **RNE 丸めと全 242 本・全 5,680,910,336 要素でビット一致**（切り捨てとは 0 本一致）。f32 に対する最大相対 RMS は 0.00179（bf16 の丸めの床）。
  §2.1 と合わせると、Wan2.2 TI2V-5B Diffusers の umT5 = google/umt5-xxl の encoder の bf16 RNE。
- 公式 `.pth` の値は読んでいない（2 リポで同じファイルであることまでが事実）。

## 7. 未確定の点

- 公式の `models_t5_umt5-xxl-enc-bf16.pth` が google の f32 の RNE 丸めと同じ値か（読んでいない）。
- google の safetensors PR 2 本で shard 1 の sha256 が違う理由（PR を使わない推奨なので未調査）。
- 容器を実際に書いて part 0 以外がバイト同一になるか（§4.2 は推測。書き手の実行は調査の境界の外）。
- `.bin` の `weights_only=True` 読みが将来の torch で警告・挙動変更を受けないか（torch 2.13.0 では警告なしで読めた — 観測のみ）。

## 出典

- HF API（一次）: <https://huggingface.co/api/models/google/umt5-xxl?blobs=true>・`/refs?include_prs=1`・`/discussions`・
  `/revision/538e820751548eeb0cdb6dd38192522d9530846c?blobs=true`・`/revision/ae98d537332051eccb6dbc3a577b9f2c5d2364df?blobs=true`・
  <https://huggingface.co/api/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers?blobs=true>・<https://huggingface.co/api/models/Wan-AI/Wan2.2-TI2V-5B?blobs=true>・
  <https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B?blobs=true>・<https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers?blobs=true>
- 取得したファイル（一次）: google の `config.json` / `pytorch_model.bin.index.json` / `README.md` / トークナイザ 4 本 / shard 1〜3
  （HF キャッシュ `models--google--umt5-xxl`）・Wan2.2 TI2V-5B Diffusers の `text_encoder/` と `tokenizer/`（`models--Wan-AI--Wan2.2-TI2V-5B-Diffusers`）・
  Wan 公式 Wan2.1-T2V-1.3B の `google/umt5-xxl/{tokenizer.json,tokenizer_config.json,special_tokens_map.json}`（作業ディレクトリ）・
  Wan2.2 Diffusers の safetensors ヘッダ（Range 取得）
- 実行したスクリプト（作業ディレクトリ `/tmp/claude-1000/recon-2026-10-04/umt5-upstream/`・`cd tools && uv run --group wan --inexact python <script>`）:
  `wan21_bf16_check.py`（§2.2）・`cfg_check.py`（§1.1）・`tok_compare.py`（§3.2）・`weights_compare.py`（§2.1 — 結果 `weights_result.json`）・
  `w22_compare.py`（§6 — 結果 `w22_result.json`）・`quant_compare.py`（§4.1・§5.1 の `checkpoint_keys`）・`trace_compare.py`（§4.1）・`fp_check.py`（§1.1）
- 前回の調査: `docs/research/2026-10-03-umt5-encoder-recon.md` §1.4（`spiece.model` 同一・`tokenizer.json` のスコア差 3.6e-15 — 今回の値と合う）

## 独立検証（2026-10-04）

別のレッグが、設計が依存する主張 14 個を一次ソースと再計測で確かめた。成り立つ（holds）が 14 個。

### 本文が落としていた、設計に効く事実

- コミュニティの追加学習版 umT5（9202c70d）は BF16 の単一 safetensors（242 キー、HF 命名）で、fp8_scaled 版もある。license と base_model は未宣言。Checkpoint の F32 だけ受ける MUST に当たるので、umT5 を差し替える仕組みには bf16 を入力にする経路と出所の扱いが要る
- Wan2.2 TI2V-5B から越境参照するとき、トークナイザ資産は karume-wan2.1 リポの部品（UMT5_ASSETS={}）。umT5 容器だけでなく資産の置き場（Wan2.1 から越境参照するか、焼き直すか）も決める必要がある
- §4.2 の表は「part 0（model-00001）」と「karume.json の part 1 の sha256」が同じファイルを指していて、0 始まりと 1 始まりの番号が混ざっている。重みの part の範囲「2〜26」も 1 始まり。移行の検証手順で取り違えるおそれがある
- write（check なし）は系列の部品ディレクトリを丸ごと差し替え、golden も消す（umt5_export.py:567-590）。そのため移行は write→reference→dist の再生成が必須で、part0Sha256 だけを書き換えては済まない
- umT5 の重みは encoder 用の shard 1〜3（約 29.8 GB）の .bin を取得する。Wan2.2 Diffusers の bf16 なら 11.4 GB で済むが、丸めの出発点が f32 でなくなり Checkpoint の前提が崩れるので、取得量と正確さのトレードオフになる
