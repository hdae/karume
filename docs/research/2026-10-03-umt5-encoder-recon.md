> この文書は 2026-10-03 時点の調査記録で、正本ではない。

# umT5-XXL のテキストエンコーダを GPU で回す — ADR 起草材料（recon）

ADR 0118 の段 10（umT5 を GPU で回して自由なプロンプトを受ける・別 ADR）の材料。読み取りと CPU の小実験だけ
（HEAD `e2b14aa9`・GPU 不使用・実重みは読んでいない — config・トークナイザ・safetensors のヘッダだけ）。
「事実」は file:line・コマンド出力・URL を根拠に持ち、「推測」は明示する。引用の行番号は HEAD のもの
（調査中に別の作業が `wan/` の recipe と `packages/models/src/wan/` を編集していて、作業ツリーでは行がずれる）。
使った版: transformers 5.14.1・diffusers 0.39.0・ftfy 6.3.1・torch 2.13.0+cpu・tokenizers 0.22.2（recipes の venv）。
一時スクリプトは scratchpad（git 追跡外）。

## 要約

- 規模: 5,680,910,336 パラメータ（語彙埋め込み 10.5 億 + 24 層 46.3 億）。i8 per-channel 5.30 GiB・i4 g32 3.31 GiB。B570 の天井
  9,600 MiB に単独で入るが DiT 段（7.31 GiB）とは同居できず、text 段を DiT 段の前に畳む順次化が前提。
- op は既存の語彙で書ける見込み（相対位置バイアスは分解経路・`gelu_new` の手書き式だけ要対処）。bucket 表は f64 と torch の f32 で全域一致。
- 新事実 1: pin した transformers 5.14.1 は語彙外の文字を id 2（umT5 では `<s>`）にする。tokenizer.json と 4.57.1 は id 3（`<unk>`）。
  固定 4 プロンプトは両経路で同じ id 列。1 文字として引ける割り当て済みの文字は 19,848 個。
- 新事実 2: ftfy は公式 negative の全角読点 `，` を `,` に変える。`&` と mojibake 由来の 449 文字を含まない乱択 20 万本では、
  決定的な変換だけの鏡像が `prompt_clean` と全件一致した。
- 推奨の骨格: anima の T5 トークナイザを共通層へ広げて再利用・語彙外は fail loudly・前処理は表で移植して mojibake と entity 候補は
  fail loudly・数値の門は「fake-quant 重みの層逐次参照に対する移植の門」と「bf16 資産・f32 参照との品質の記録 + 動画の目視」の二層。

## 1. upstream の構成

### 1.1 config（`text_encoder/config.json`・pin した revision `0fad780a…`）

| 項目                 | 値                                                                                                           | 根拠・備考                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| クラス               | `UMT5EncoderModel`（`_name_or_path` = `google/umt5-xxl`）                                                    | config.json・`model_index.json` の `text_encoder`                                      |
| 層数                 | 24（`num_layers`）                                                                                           | config.json                                                                            |
| `d_model` / `d_ff`   | 4,096 / 10,240                                                                                               | config.json                                                                            |
| heads × `d_kv`       | 64 × 64                                                                                                      | config.json                                                                            |
| FFN                  | gated（`wi_0`・`wi_1`・`wo`・bias なし）・活性 `gelu_new`                                                    | config.json `feed_forward_proj: gated-gelu`・`modeling_umt5.py:103-130`                |
| `gelu_new`           | `0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))` の手書き式                                                          | `transformers/activations.py:59-66`                                                    |
| 正規化               | 平均を引かない RMSNorm・eps 1e-6・分散は f32 で計算                                                          | `modeling_umt5.py:54-77`                                                               |
| attention のスケール | 無し（`QKᵀ` にそのまま bias を足す）                                                                         | `modeling_umt5.py:298-317`                                                             |
| softmax              | f32 で計算して元の dtype へ戻す                                                                              | `modeling_umt5.py:317`                                                                 |
| 相対位置バイアス     | **層ごとに別**（全 self-attn 層が `has_relative_attention_bias=True`）・32 buckets・max_distance 128・双方向 | `modeling_umt5.py:329-332`・ヘッダで `relative_attention_bias.weight [32,64]` が 24 本 |
| `scalable_attention` | config に `true` があるが、モデリングコードは参照しない                                                      | `rg scalable` が `models/umt5/` で 0 件                                                |
| 語彙埋め込み         | `shared.weight [256384, 4096]`（encoder の `embed_tokens` と共有・倍率なし）                                 | ヘッダ・`modeling_umt5.py:1146-1160`                                                   |
| `vocab_size`         | 256,384（トークナイザの語彙 256,300 + 使われない 84 行）                                                     | config.json・tokenizer.json                                                            |
| pad / eos            | pad 0・eos 1（`decoder_start_token_id` 0）                                                                   | config.json                                                                            |
| 格納                 | F32 safetensors 5 分割・`total_size` 22,723,641,344 B                                                        | `model.safetensors.index.json`                                                         |
| fp16 のクランプ      | 各残差の後に inf を `finfo.max−1000` へ（f16 のときだけ）                                                    | `modeling_umt5.py:409-438`                                                             |

公式の実装（`wan/modules/t5.py`）も同じ構成（前回調査 `2026-10-02-video-gen-recon.md` §3.7 と一致 — 今回は HF 版を実物で確認）。

### 1.2 パラメータの内訳と格納型ごとの重み

safetensors のヘッダを数えた値（実重みは読んでいない）。

| 部分                | パラメータ数  | 内訳                                                                                      |
| ------------------- | ------------- | ----------------------------------------------------------------------------------------- |
| 語彙埋め込み        | 1,050,148,864 | `[256384, 4096]`                                                                          |
| linear（24 層）     | 4,630,511,616 | 1 層 = q・k・v・o `[4096,4096]` × 4 + `wi_0`・`wi_1` `[10240,4096]` + `wo` `[4096,10240]` |
| norm と相対位置の表 | 249,856       | norm 2 本 × 24 + 最後の norm + `[32,64]` × 24                                             |
| 計                  | 5,680,910,336 | 前回調査の値と一致                                                                        |

格納型ごとのバイト数（計算。i8 は per-channel の F32 scale〈linear は出力チャネル・embedding は行 — `names.ts:388-405`〉を含む。
i4 は group 32・scale は F32〈`quantize.py:25`・`:33`〉。norm と `[32,64]` の表は F32 のまま）:

| 格納                 | 全体                          | 埋め込みを除く（ホスト gather 時） | 埋め込みだけ                 |
| -------------------- | ----------------------------- | ---------------------------------- | ---------------------------- |
| bf16（上流の実行形） | 11,361,820,672 B（10.58 GiB） | —                                  | —                            |
| i8 per-channel       | 5,686,617,600 B（5.30 GiB）   | 4,635,443,200 B（4.32 GiB）        | 1,051,174,400 B（1,002 MiB） |
| i4 g32               | 3,551,412,224 B（3.31 GiB）   | 2,895,069,184 B（2.70 GiB）        | 656,343,040 B（626 MiB）     |
| f16（参考）          | —                             | —                                  | 2,100,297,728 B（2.0 GiB）   |

注意: `[32,64]` の相対位置の表は embedding op で引くことになる（§4.3）。exporter の量子化が embedding を対象に含める規則なら、
この小さい表まで i8 になりうる（推測 — 量子化の対象選択の規則は未確認）。表は 24 × 8 KiB なので F32 のままにする指定が要る。

### 1.3 活性の大きさと計算量（計算）

| 量                                      | L = 126（negative の長さ） | L = 512 |
| --------------------------------------- | -------------------------- | ------- |
| 残差 `[L,4096]` f32                     | 1.97 MiB                   | 8 MiB   |
| FFN の中間 `[L,10240]` f32（1 本）      | 4.92 MiB                   | 20 MiB  |
| スコア・バイアス `[64,L,L]` f32（1 本） | 3.88 MiB                   | 64 MiB  |
| MAC（24 層）                            | 0.587 T                    | 2.42 T  |

512 トークンでも 1 層の中間は数百 MiB 以下で、重み（i8 で 5.30 GiB）が支配する。

### 1.4 トークナイザ — 3 つの定義が食い違う

同じ `spiece.model` から、3 通りの実挙動がある。

| 項目         | `spiece.model`（SentencePiece の原本）                         | `tokenizer.json`（transformers 4.x が読む）        | transformers 5.14.1 の実挙動（karume の pin）        |
| ------------ | -------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| モデル       | Unigram（`model_type` 1）・256,000 ピース                      | Unigram・256,300（+ `<extra_id_*>` 300）           | Unigram・256,300（語彙は tokenizer.json と同一）     |
| 正規化       | `identity`・`add_dummy_prefix` 1・`remove_extra_whitespaces` 0 | `Replace(" {2,}" → " ")` だけ                      | **無し**                                             |
| pre-tokenize | （SentencePiece 内部）                                         | `Metaspace`（prepend always・split）だけ           | **`WhitespaceSplit` + `Metaspace`**                  |
| 語彙外の文字 | **byte_fallback 1**（trainer の field 35）                     | unk_id **3**（`<unk>`）・byte_fallback false       | unk_id **2**（umT5 では `<s>`）・byte_fallback false |
| 後処理       | —                                                              | `A + </s>`                                         | 同左                                                 |
| added tokens | —                                                              | 304（`<pad>` `</s>` `<s>` `<unk>` + extra_id 300） | 同左                                                 |

- ピースの種別（protobuf を手で解いた値）: NORMAL 255,727・UNKNOWN 1・CONTROL 3・USER_DEFINED 13（`[eod]` `[web]` など）・BYTE 256。
  user-defined の 13 個は tokenizer.json では added token ではなく普通の語彙（`[web]` は id 5 の 1 トークンになる — 実測）。
- transformers 5.14.1 の組み直しの出所: `transformers/models/t5/tokenization_t5.py:112-127`。`Unigram(..., unk_id=2,
  byte_fallback=False)` を固定で書き、`WhitespaceSplit` + `Metaspace` を差す。コメントは「T5 vocab structure: <pad>=0, </s>=1,
  <unk>=2」（`:99`）で、元の T5 の語彙配置を前提にしている。umT5 は `<s>`=2・`<unk>`=3（tokenizer.json の先頭 4 行）。
  `tok.unk_token_id` は 3 を返すが、Unigram の中の unk は 2（`tok._tokenizer.to_str()` の `model.unk_id`）。
- 実測（transformers 5.14.1 の `AutoTokenizer`）: `𠀋𡈽` → `[273, 2, 1]`（`▁` `<s>` `</s>`）。同じ入力を `tokenizers.Tokenizer.from_file`
  （tokenizer.json）で → `[273, 3, 1]`。transformers 4.57.1（一時の隔離環境・`T5TokenizerFast`）でも `[273, 3, 1]`・unk_id 3。
- 公式 Wan2.1 は `AutoTokenizer.from_pretrained` で読み（`wan/modules/tokenizers.py:44`）、要求は `transformers>=4.49.0`
  （上限なし — `requirements.txt`）。4.x では tokenizer.json の挙動（id 3）になる（4.57.1 で確認。4.49.0 自体は未確認）。
- 全コードポイントの掃引（サロゲートを除く 1,112,064 個を `"x" + c + "y"` で符号化）:
  - 語彙外（transformers 5 の経路で id 2 が出る）: 1,092,156 個。割り当て済み（Python の UCD 16.0.0）292,531 個のうち、
    1 文字として引けるのは 19,848 個（Lo 13,502・So 1,759・Co〈私用領域〉1,094・Ll 1,035 など・U+10000 以上 1,441 個）。
  - 2 経路の差は「語彙外の id（2 か 3）」と、空白 24 文字（`regex` の `\s` の 25 文字から U+0020 を除いたもの）の分割だけ。
    後者は `prompt_clean` が先に U+0020 1 個へ潰すので、正規化後の入力では効かない。
- 固定 4 プロンプト（正規化後）は 2 経路で id 列が同一・語彙外なし（長さ 28 / 118 / 50 / 126 — ADR 0118 段 6 の値と一致）。
  よって今の埋め込み資産はこの食い違いの影響を受けていない。
- google/umt5-xxl との関係: `spiece.model` は同一（sha256 `e3909a67…`・4,548,313 B）。tokenizer.json は別ファイル（sha256 `af904105…` vs
  Wan-AI 版 `20a46ac2…`）だが、ピースは全件同一・スコアの差は最大 3.6e-15（浮動小数の書き出し桁）・pre-tokenizer は旧書式
  `add_prefix_space: true` と新書式の同義表現。ライセンスは HF の cardData で `apache-2.0`（前回調査 §3.10 の未確認を解消）。
- 推測: umT5 の事前学習は byte_fallback ありの SentencePiece で回ったので、語彙外の文字を `<unk>` / `<s>` の 1 トークンにした入力は、
  id 2・3 のどちらでも学習時に見なかった分布になる。Wan の学習時の id は transformers の版に依存し、確かめる手段が無い。

## 2. `_get_t5_prompt_embeds` の意味論と今の資産の写し方

### 2.1 上流（diffusers 0.39.0 `pipelines/wan/pipeline_wan.py`）

- `prompt_clean` を掛ける（`:170`・定義は `:78-93`）→ トークナイザ `padding="max_length"`・`max_length`・`truncation=True`・
  `add_special_tokens=True`・`return_attention_mask=True`（`:173-181`）。`__call__` の既定 `max_sequence_length` は 512（`:403`）。
  512 を超える入力は**黙って切り詰める**（末尾に `</s>` を残す）。
- `seq_lens = mask.gt(0).sum(dim=1)`（`:183`）。encoder にマスクを渡す（`:185`）。マスクは `(1−m)·finfo.min` の加算型
  （`modeling_umt5.py:674-677`）で、全層のバイアスに足される（`:310-311`）。
- 出力を `dtype` に変換（`:186`）→ 有効長で切り（`:187`）→ `new_zeros` で 512 行へ詰め戻す（`:188-190`）。dtype の既定は
  `text_encoder.dtype`（`:167`）で、公式の構成では bf16。DiT へ渡す前に transformer の dtype へ変換する（`:547-550`）。

### 2.2 今の recipe（`tools/export-recipes/wan/text_embeds.py`）

- 上流の `_get_t5_prompt_embeds` を**そのまま呼び**、`[1,512,4096]` を有効長で切って f32 で資産に書く（`:154-184`）。式は写していない。
- umT5 は bf16・CPU（`:86`・`:116-127`）。出力の bf16 → f32 は値を変えない（実測: 4 本とも全要素の下位 16 ビットが 0）。
- `valid_length` は 512 超を fail loudly（上流は黙って切る — `:130-151`）。有効長の後ろが厳密にゼロ・最後の行が非ゼロを検査（`:177-180`）。
- `normalize` は ftfy が import できないと止める（上流は黙って `fix_text` を飛ばす — `:101-113`）。正規化前後の文字列をメタに書く（`:221-233`）。
- 資産の値（実測）: 出力の rms 0.073〜0.084・最大絶対値 1.09〜1.66・行ノルム 0.63〜7.30。資産 5,280,288 B（sha256 先頭 `685f5cbf63ae22f7`）。

### 2.3 有効長だけで回してよいか

- 導出（前回調査 §3.7）: パディング位置のキーはマスクで重み 0、バイアスは距離だけで決まり、出力は有効長で切られる。
- 小実験（実重みではない — 4 層・d_model 64 の乱数初期化 UMT5、`L` = 1 / 2 / 28 / 126 / 300 / 511 / 512）:
  「512 に詰めてマスク」と「有効長だけ・マスク全 1」の差は f32 で最大相対 5.7e-7（7 本中 4 本がビット一致・差は GEMM の縮約順）、
  bf16 では 7 本ともビット一致。実モデル（24 層・K = 10,240）での一致は未確認。
- 帰結（推測）: GPU 側は有効長 `L` の動的形で回し、マスク入力を持たない形にできる。計算量は `L` に比例して減る（negative の 126 なら
  512 の 24%）。

## 3. プロンプトの前処理（`prompt_clean`）

### 3.1 定義

- diffusers: `basic_clean` = ftfy が入っていれば `ftfy.fix_text` → `html.unescape` を 2 回 → `strip()`。`whitespace_clean` = `regex` の
  `\s+` を U+0020 1 個へ → `strip()`。`prompt_clean` = `whitespace_clean(basic_clean(text))`（`pipeline_wan.py:78-93`）。
- 公式 Wan2.1: 同じ式だが ftfy は**無条件**（`wan/modules/tokenizers.py:12-21`）。diffusers だけが「ftfy が無ければ飛ばす」分岐を持つ。
- `regex` の `\s` は 25 文字（U+0009〜000D・0020・0085・00A0・1680・2000〜200A・2028・2029・202F・205F・3000 — 実測）。
  Python の `str.strip()`（`isspace`）はさらに U+001C〜001F を落とすが、この 4 文字は ftfy の `remove_control_chars` が先に消す。

### 3.2 ftfy 6.3.1 の `fix_text`（既定の設定）

設定の既定値は `ftfy/__init__.py:212-227`。`fix_text` は入力を改行ごとの区切りに分け、区切りに `<` があればその区切りだけ
`unescape_html` を切る（`:330-362`）。各区切りに `fix_and_explain` を**不動点まで繰り返す**（`:364-423`）。1 周の順序と中身:

| 順 | 処理                      | 何を変えるか（`ftfy/fixes.py`）                                                                | 決定的か                                                     |
| -- | ------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 1  | `unescape_html`           | `&名前;` / `&#数字;` の entity（大文字の Latin 名も一部）を文字へ（`:94-140`）                 | 表引き（HTML5 の entity 表）                                 |
| 2  | `fix_encoding`            | mojibake の修復。ASCII だけの文字列と `is_bad` が偽の文字列はそのまま（`__init__.py:482-483`） | **ヒューリスティック**（正規表現 4,050 文字 + 複数の codec） |
| 3  | `fix_c1_controls`         | U+0080〜009F を Windows-1252 として読み直す                                                    | 表引き                                                       |
| 4  | `fix_latin_ligatures`     | `ﬁ` → `fi` など Latin の合字だけ（`:168-185`）                                                 | 表引き                                                       |
| 5  | `fix_character_width`     | 全角・半角の形を標準形へ・U+3000 → U+0020（`:186-205`）                                        | 表引き                                                       |
| 6  | `uncurl_quotes`           | 曲がった引用符を `'` `"` へ（`:158-166`）                                                      | 表引き                                                       |
| 7  | `fix_line_breaks`         | CRLF・CR・U+2028・U+2029・U+0085 → `\n`（`:206-256`）                                          | 表引き                                                       |
| 8  | `fix_surrogates`          | 対のサロゲートを 1 文字へ・孤立は U+FFFD（`:274-295`）                                         | 表引き                                                       |
| 9  | `remove_terminal_escapes` | `ESC [ … 英字` を削る（`:142-156`）                                                            | 正規表現                                                     |
| 10 | `remove_control_chars`    | U+0000〜0008・000B・000E〜001F・007F・206A〜206F・FFF9〜FFFC・FEFF などを削る（`:297-329`）    | 表引き                                                       |
| 11 | 正規化                    | NFC（`__init__.py:225`）                                                                       | Unicode の版に依存                                           |

実測（`ftfy.fix_text`）:

- 公式 negative プロンプトは `fix_character_width` で U+FF0C `，` が `,` に変わる（変化はこの 1 種類だけ — `fix_and_explain` の説明は
  `[('apply', 'fix_character_width')]`）。正 3 本は不変。
- `“quoted”` → `"quoted"`・`ﬁne` → `fine`・`ＡＢＣ` → `ABC`・`ｶﾀｶﾅ` → `カタカナ`・`\x1b[31mred` → `red`・`a\x07b` → `ab`・
  `cafÃ©` → `café`（mojibake の修復）・`﻿bom` → `bom`。`x​y`（ゼロ幅空白）は不変。
- `is_bad` は `café`・`naïve résumé`・`it’s a dog’s life`・`Zürich Straße`・`São Paulo à noite`・キリル・ギリシャ・`½ cup` で偽、
  `Ã©t`・`â€œquotedâ€`・`naÃ¯ve` で真。

### 3.3 mojibake 判定が触る文字（構文木の抽出 + fuzz）

- `BADNESS_RE` の構文木から字面の文字と範囲を集めると、非 ASCII は 449 文字（C1 の 32 文字を含む・Latin 131・Cyrillic 97・Greek 65・
  罫線 37 など）。CJK・仮名・ハングルは 0 文字。ただし正規表現は `.`（8 箇所）・`\w`・`[^A-Za-z]` も使うので、
  「449 文字を含まなければ偽」は構文からは言えない。
- fuzz（乱数 seed 0）: 「`&` と 449 文字を含まない」文字の池（ASCII 印字可能・制御文字・CJK・仮名・CJK 記号・ハングル・全角半角形・
  合字・曲がった引用符・結合文字・絵文字・アラビア・デーヴァナーガリー・タイ・キリルとギリシャと Latin-1 補助のうち 449 外 — 計 4,220 文字）
  から長さ 1〜40 を 200,000 本: `is_bad` が真 0 本・「表引きの 9 処理 + NFC を不動点まで → strip → `\s+` 畳み → strip」の鏡像と
  `prompt_clean` の不一致 0 本。449 文字も混ぜた池（4,558 文字）の 50,000 本では不一致 277 本（0.55%）。
- 帰結: 「`&` と mojibake 判定の対象文字を含まない入力」に限れば、ヒューリスティック（`fix_encoding`）と entity 表を移植せずに
  表引きだけで `prompt_clean` と一致する（fuzz の範囲での事実・網羅ではない）。

### 3.4 TS で同じ結果を保証する選択肢

| 案                                               | 中身                                                                                                                                                                                                                      | 守るもの                                                                                                                  | 捨てるもの・リスク                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) ftfy を全部移植                              | `fix_encoding`（`is_bad` + codec 群〈sloppy-windows-1252 など〉+ `restore_byte_a0` 等）・entity 表・`html.unescape`（Python 版の entity 2,231 名・セミコロン無しの最長一致・数値参照の置換表）・NFC 表                    | 上流と同じ受理集合（何でも受ける）                                                                                        | 移植量が最大（ftfy の本体だけで 2,731 行 + `bad_codecs`）。ヒューリスティックの parity は fuzz でしか縛れない。版を上げるたびに追従が要る                                                                                                                                                                                                                      |
| (b) 変化しない入力だけ受理                       | 「`prompt_clean(x) == x` でない」入力は拒む。判定に要るのは結局 (b') と同じ表                                                                                                                                             | 実装量が最小・沈黙の誤変換が無い                                                                                          | 公式 negative（全角読点）・曲がった引用符・全角英数・改行を含む普通の入力まで拒む。実用の受理集合が狭すぎる                                                                                                                                                                                                                                                    |
| (b') 決定的な処理だけ移植し、残りは fail loudly  | 表 9 本 + NFC を不動点まで回し、その後 strip / 空白畳み。`is_bad` が真になる入力（`BADNESS_RE` を移植）と HTML の entity 候補（`&` の後に空白・`<`・`&` 以外が続く）は拒む。C1 文字（U+0080〜009F）と孤立サロゲートも拒む | 普通の入力（欧文のアクセント・CJK・全角・引用符・改行）を上流と同じ文字列にする。ヒューリスティックの修復は黙ってやらない | `cafÃ©` のような mojibake と `&amp;` を含む入力は受けない（利用者が直してから渡す）。`BADNESS_RE` の JS への翻訳（`\w` が Unicode 対応か・`.` の改行）を parity で縛る必要がある。NFC は JS の `normalize` と Python の UCD の版差を検査する必要がある（anima でも素の `normalize("NFC")` では再現できない例があり、表を焼いた — `anima_tokenizer_test.ts:6`） |
| (c) 正規化は利用者側・資産に `normalized` を記録 | 今の方式の延長。GPU 経路は「正規化済みの文字列」だけを受け、ホストは `prompt_clean(x) == x` を満たさない入力を拒む                                                                                                        | 実装は (b) と同じで最小                                                                                                   | 正規化の責任が利用者に移る（ブラウザの利用者は Python を持たない）。受理集合は (b) と同じで狭い                                                                                                                                                                                                                                                                |

推奨（推測を含む）: **(b')**。理由は (1) 決定的な処理は表と NFC だけで、表は recipe が ftfy から焼いて全件・乱択で鏡像と突き合わせられる
（anima の `text.py` の正規化表と同じ流儀 — `tools/export-recipes/anima/text.py:1-20`）(2) ヒューリスティックは「黙って近似しない」
（CLAUDE.md の横断の不変条件）に従い、真になったら拒むだけにできる (3) (b) / (c) の受理集合では公式 negative すら素のまま受けられない。
段の分け方としては、最初は「`&` と 449 文字を含めば拒む」の強い版で出し、`is_bad` の移植は後の段で受理集合を広げる形にもできる
（強い版は fuzz で鏡像と全件一致した範囲そのもの）。ただし強い版はアクセント付きの欧文と `’` を拒む。

## 4. karume の再利用点

### 4.1 トークナイザ（`packages/models/src/anima/text/`）

anima の T5 は Anima の `t5_tokenizer`（語彙 32,100・added 103・unk_id 2・`Precompiled` 正規化・`WhitespaceSplit` + `Metaspace`）を写したもの
（`t5-tokenizer.ts:1-15`・HF キャッシュの `circlestone-labs/Anima-Base-v1.0-Diffusers` の `t5_tokenizer/tokenizer.json` を実測）。
Unigram の本体（Viterbi・同点処理・`fuse_unk`・任意の byte_fallback）は既に家族非依存の `packages/models/src/text/unigram.ts` にある
（`:1-80`）。umT5 に対して:

| 部品         | anima の実装                                                                                                                      | umT5 に要るもの                                                                                                                              | 差                                  |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| Unigram      | `src/text/unigram.ts`（共有済み）                                                                                                 | 同じ。語彙 256,300・最長ピース 16 コードポイント・最小スコア −19.383                                                                         | 無し                                |
| added tokens | `src/text/added-tokens.ts`（leftmost-longest）                                                                                    | 304 個（`<pad>` `</s>` `<s>` `<unk>` + extra_id 300）。`a </s> b` は `[289, 1, 748, 1]` になる（実測 — 特殊トークンが本文から入る）          | 表が違うだけ                        |
| 正規化       | `Precompiled` を焼いた表（`spm-normalizer.ts`）が**必須**（`T5Assets.normalizer`）                                                | transformers 5 の経路は正規化**無し**（tokenizer.json の経路でも `prompt_clean` 後は効かない）                                               | `T5Assets` の正規化を省略可能にする |
| pre-tokenize | `t5PreTokenize`（`WhitespaceSplit` → `Metaspace` split・`t5-tokenizer.ts:53-68`）                                                 | transformers 5 の経路と同じ形。空白集合は Rust の `char::is_whitespace`（anima は Qwen2 の `\s` と一致を emit 時に検査 — `text.py:419-432`） | 無し（表を流用できる見込み）        |
| 語彙外       | unk 1 個（`fuse_unk`）                                                                                                            | §6 の軸 A（id 2 / id 3 / 拒否）                                                                                                              | 方針次第                            |
| 切り詰め     | `maxLength − 1` まで積んで `</s>`（`t5-tokenizer.ts:117-136`）                                                                    | 上流 `truncation=True`・`max_length=512` と同じ形                                                                                            | 無し（512 超を拒むなら別の門）      |
| 資産の形     | exporter が語彙を引くだけの JSON に焼く（`tokenizer.ts:1-12`・anima は 2 本で計 4.6 MB）                                          | 語彙とスコアの JSON は 8,576,480 B（gzip 3,318,348 B — 計算）。anima の T5 の同じ部分は 1,059,227 B                                          | サイズ 8 倍                         |
| テスト       | `anima_tokenizer_test.ts` が recipe の生成する parity fixture（id 列 + 語彙の部分集合 + 正規化表の全体）と突き合わせる（`:1-15`） | 同じ形。fixture の正本を transformers 5 にするか tokenizer.json にするかが軸 A                                                               | —                                   |

推測: T5 トークナイザを anima のディレクトリから `src/text/` へ一般化（正規化を省略可能に・表を引数に）すれば、umT5 は表を差し替えるだけで
済む。anima 側の parity fixture は動かない（同じ入力で同じ id 列 — 振る舞いの検査）。

### 4.2 exporter での op の拾われ方

| umT5 の部品                 | export で出る形（推測を含む）                                                                 | IR の op                                                                                                                                                                     | 状態                                                                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 語彙埋め込み                | `aten.embedding`                                                                              | `embedding`（i8 / i4 の格納あり — `docs/quantization.md` 表 1）                                                                                                              | ある                                                                                                                                                   |
| q / k / v / o・FFN          | `aten.linear` 相当                                                                            | `linear`（i8 per-channel・i4 g32）                                                                                                                                           | ある                                                                                                                                                   |
| RMSNorm                     | `to(f32)` → `pow(2)` → `mean` → `+eps` → `rsqrt` → `mul` → `mul(weight)`                      | `rms_norm`（`normalize._fold_rms_norm` が pow/mean/rsqrt 形を畳む — `normalize.py:325-356`）                                                                                 | ある（畳まれるかは export で確認 — 推測）                                                                                                              |
| `gelu_new`                  | 手書き式。`pow(x, 3)`・`tanh`・`mul`・`add` に降りる                                          | `gelu_tanh` は `aten.gelu(approximate="tanh")` からしか出ない（`aten_handlers.py:114-122`）。`pow(x,3)` は畳む規則が無い（`_pow2_to_mul` は指数 2 だけ — `normalize.py:29`） | **足りない**: recipe で活性を `nn.GELU(approximate="tanh")` に差し替えるか、`pow(x,3)` の正規化を足す。どちらも丸めが上流の eager と変わりうる（推測） |
| attention                   | `matmul(q, kᵀ)` → `+ bias` → `softmax(float)` → `matmul(·, v)`（スケール無し・SDPA ではない） | `bmm`（または `matmul`）・`add`・`softmax`                                                                                                                                   | ある。融合 attention は mask が `[1,1,M,N]` 限定で、head ごとのバイアス `[1,64,L,L]` は入らない（`names.ts:216-243`）                                  |
| 相対位置のバケット          | `arange` → 差 → `abs`・`log` → `where` → `min`（`modeling_umt5.py:189-246`）                  | §4.3 の通り、グラフの外（ホスト）で作るのが前例                                                                                                                              | —                                                                                                                                                      |
| 層ごとの表 `[32,64]` の引き | `aten.embedding(table, bucket[L,L])` → `[L,L,64]` → `permute`                                 | `embedding`（添字は rank 1 以上なら可 — `shapes.ts:951-964`）・`permute`                                                                                                     | ある                                                                                                                                                   |

### 4.3 相対位置のバケット表

- 実測: `_relative_position_bucket` を torch（f32 の `log`）で距離 −511〜511 に当てた値と、f64 の `math.log` で同じ式を計算した値は全件一致。
  境界は距離 0〜7 がそのまま、8・12・16・23・32・46・64・91 で次のバケットへ移る（負の側は 0〜15、正の側は +16 で 17〜31・バケット 16 は未使用）。
- 前例: SBV2 の DeBERTa は相対位置の添字表をホストで作ってグラフ入力で受け、Python 生成器と TS 生成器のバイト一致を別テストで縛る
  （`packages/models/tests/sbv2_rel_pos_parity_test.ts:1-11`）。Anima の RoPE 表も「入力昇格」で、S に依存する焼き込み定数をゼロにした
  （ADR 0034 決定 2 — `0034-dit-dynamic-tokens.md:26-32`）。
- ADR 0118 決定 4 の今の文面は「i32 のバケット添字表 `[512,512]` = 1 MiB の定数 + 層ごとの表の gather」（`0118-wan21-video-generation.md:365-368`）。
  動的 `L` で回すなら、定数案は `[:L,:L]` の切り出し（`sym_prefix_slice`）が要る。入力案は前例どおりで、f64 と f32 の一致は上の実測が支える。

### 4.4 Session の admission と e2e の数値の門（前例）

- Wan の Session は段ごとに張って畳む（DiT の段で `createSession`・終わりで dispose — `packages/models/src/wan/pipeline.ts:933`〈HEAD の
  行〉付近）。家族の admission は重みの part を取る前に manifest・quant・Session の宣言を検査する（`pipeline.ts` の `#admit`）。text 段は
  この並びの先頭に足す形になる（推測）。
- anima の text_encoder（Qwen3）は f16 の格納（`tools/export-recipes/anima/distribution.py:386`）で、e2e は最終 PNG の sha256 だけ
  （tolerance 化は禁止 — `packages/models/tests/e2e_anima_test.ts:13`）。段ごとの数値の帯は持たない。
- DeBERTa（sbv2 の BERT・24 層の encoder）の i8 は、golden を **fake-quant 後の重み**で torch から採り、門は実測から導いた atol
  （i8 系列 7e-4 = 素の最悪 1.23e-4 の 5.7 倍）。元のモデルとの差は層別 SNR（約 1 dB/層で単調に下がる）と聴感の利用者裁定で別に見た
  （ADR 0026 決定 2・4・`0026-w8a8-deberta-deployment.md:30-85`）。活性まで量子化すると 1 層目の出口にしか判別帯が無かった（同 `:78-85`）。
- Wan の DiT は f64 の参照に対する正規化比 r の帯（段 3: 帯 75・段 8: 帯 104 = 決定用の最悪 × 5・受入れは別ケース）で縛った
  （ADR 0118 検収表・backlog の Wan 行）。

## 5. 容量の見立て

| 項目                          | 値                                                                                                                                   | 根拠                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| B570 の天井（Deno の総確保）  | 9,600〜9,631 MiB（保守側 9,600 MiB）                                                                                                 | `2026-10-02-b570-vram-budget.md` §3-2 |
| DiT 段の山                    | 33 フレーム 6.19 GiB（モデルカードの値）・81 フレーム 7.31 GiB                                                                       | ADR 0118 `:850-851`・段 8 の結果      |
| VAE 段の山                    | 3.32 GiB                                                                                                                             | ADR 0118 段 6 の結果                  |
| text 段（i8・埋め込みも GPU） | 重み 5.30 GiB + 活性 数百 MiB 以下 + staging（推測）                                                                                 | §1.2・§1.3                            |
| text 段（i4 g32）             | 重み 3.31 GiB + 同上                                                                                                                 | §1.2                                  |
| text 段と DiT 段の同居        | i8: 5.30 + 7.31 = 12.6 GiB > 9.38 GiB・i4: 3.31 + 7.31 = 10.6 GiB > 9.38 GiB（81 フレーム）。33 フレームでも i8 は 11.5 GiB で越える | 計算                                  |
| 束縛上限（B570・Deno）        | `maxStorageBufferBindingSize` 2,147,483,644 B。i8 の語彙埋め込み 1,051,174,400 B は入る                                              | ADR 0118 `:81-83`                     |
| ブラウザ                      | WebGPU 既定の束縛上限 128 MiB・Chromium の単一 ArrayBuffer 上限 2,145,386,496 B（段 9 で実測）                                       | ADR 0118 `:673-674`・`:850-851`       |
| 配布形の今のサイズ            | 3,136,930,382 B（2.92 GiB — `models/karume-wan2.1` のファイルの和）                                                                  | `find -printf '%s'`                   |
| 配布形 + umT5                 | i8: 約 8.2 GiB・i4: 約 6.2 GiB（トークナイザ資産 数 MB を除く）                                                                      | 計算                                  |
| part の上限                   | 1 part ≤ 1,024 MiB・part 数 ≤ 1,024（`karume/5`）                                                                                    | ADR 0109 `:89-91`                     |

- 帰結: text 段は DiT 段の**前に畳む**順次化が必須（DiT → VAE と同じ扱い）。段 6 の実測では DiT を畳んだ直後に山が下がり、
  初回だけ約 0.24 GiB が残った（未説明 — ADR 0118 追記 2026-10-02）。text → DiT の切り替えで同じ残りが出るかは未測。
- i8 の語彙埋め込み（1,002 MiB）は 1 part の上限 1,024 MiB に収まるが余裕は 22 MiB。テンソルの piece 分割（ADR 0090）で part を跨げるので
  上限には当たらない（推測 — 今の Wan の part は約 250 MB で、ミラーの part 長は 256 MiB）。
- 埋め込みをホスト gather にする（PLE の前例 — ADR 0085）と、GPU の重みは 4.32 GiB（i8）に下がり、束縛上限の問題が消える。代わりに
  ホスト RAM に 1 GB 前後を持つ（ブラウザでは単一 ArrayBuffer の上限の内）。
- export のホスト RAM: F32 の重みだけで 22.7 GB。gemma4 E4B 通常版の export は 31 GiB 機で OOM した（`docs/backlog.md:240-241`）。
  ADR 0118 は「第 2 段の最初の作業はこの実測」と書く（`0118-wan21-video-generation.md:371-375`）。

## 6. 数値の門の案

### 6.1 誤差の桁（推測を含む）

- 参照（今の資産）は bf16 の umT5 の出力。bf16 の 1 ulp は値 1〜2 で 2⁻⁷ ≈ 7.8e-3、rms 程度の値（約 0.08）で 2⁻¹¹ ≈ 4.9e-4。
  出力の丸めだけでこの半分の誤差があり、24 層を bf16 で回した累積はさらに大きい（未測）。
- i8 per-channel の重み誤差（推測）: 行の最大絶対値が分布の 4〜5σ なら、刻みは約 0.035σ・丸めの rms は約 0.01σ（重みの相対 1% 前後）。
  bf16 の重みの丸め（相対 rms 約 2e-3）より数倍大きい見込み。mlx-umt5 は per-channel int8 を「安全」と報告している（前回調査 §6・自己計測）。
- よって「GPU i8 vs bf16 資産」の差は、bf16 自身の誤差と i8 の誤差が混ざる。比べるなら f32（量子化なし）の参照を第 3 の点に置き、
  「i8 の誤差 ÷ bf16 の誤差」の比で見るのが筋（推測）。

### 6.2 門の二層（前例に合わせた案）

1. **移植の門（ビット級に近い・帯は実測導出）**: 参照 = 同じ i8 の fake-quant 重みで回した CPU の参照（f64 か f32）。DeBERTa の i8 と
   同じ「golden は fake-quant 後の重み」（ADR 0026 決定 2）、帯は DiT と同じ「決定用ケースの最悪 × 5・受入れは別ケース」。
   ホスト RAM の制約から、参照は層ごとに safetensors から読んで回す**層逐次**の形が要る（推測 — f64 の全重みは 45 GB で 31 GiB 機に載らない。
   1 層は f64 で約 1.5 GB）。
2. **品質の記録（門ではなく観測 + 利用者の目視）**: GPU i8 と bf16 資産・f32 参照の 3 点の差（行ごとのコサイン類似度・相対フロベニウス誤差・
   最大絶対値）を 4 プロンプト + 追加プロンプトで記録。最終判断は同じ seed の動画の A/B を利用者が目視（seed 4 本以上 × プロンプト 2〜3 種の
   対で渡す運用）。DeBERTa の聴感裁定（ADR 0026 決定 4）と同じ位置づけ。

### 6.3 sha 行の扱い

- 既定の経路が「事前計算の資産」から「GPU の umT5」に変わると、同じプロンプト・seed でも DiT の入力が変わり、動画の sha 行は全部変わる。
- sha 門の tolerance 化は禁止（CLAUDE.md の検証コマンド節）。新しい経路は**新しい case id** の sha 行として足し（`KARUME_REFERENCE=write` —
  ADR 0106）、資産の経路の行は資産の経路が残る限りそのまま残すのが筋（推測）。
- 参照門は「caseIds のどれか 1 本に行があれば緑」で、新しい case id の書き忘れが警告付き SKIP で通る既知の抜けがある（backlog の
  Wan 行の隣接の小物 2026-10-03）。新しい case id を足す前にこの抜けを締めるか、足した行の存在を別に確かめる必要がある。

## 7. 実装の分割案と段 9 との重なり

### 7.1 段の候補（検収の形）

| 段          | 中身                                                                                               | 検収                                                                                                                                              | GPU  |
| ----------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| 10a         | トークナイザ（T5 実装の一般化 + umT5 の表）と前処理（§3.4 の案）の TS 実装・recipe の fixture 生成 | id 列の parity（固定 4 本 + 境界ケース + 乱択）・前処理の鏡像の fuzz の parity・拒否ケース（語彙外・`&`・mojibake・空文字・512 超）の fail loudly | 不要 |
| 10b         | export のホスト RAM の実測と export の形の確定（全体 1 グラフ / 層逐次の emit / 分割）             | ピーク RSS の記録・31 GiB 機で完走するか                                                                                                          | 不要 |
| 10c         | umT5 の export（i8・活性 f32・動的 `L`・バケット表は入力）と層逐次の参照・TS のバケット表生成器    | バケット表の Python / TS のバイト一致・GPU 出力が移植の門（§6.2 の 1）に入る・故障注入（表を 1 ずらす・活性を差し替えない）が帯の外               | 要る |
| 10d         | パイプラインへの結線（text 段 → DiT 段の順次化・admission・manifest の部品・配布形とカード）       | 品質の記録（§6.2 の 2）・VRAM の山と切り替え時の残り・新しい case id の sha 行・動画の目視 A/B                                                    | 要る |
| 10e（任意） | i4 の席・埋め込みのホスト gather（Chrome の上限次第）                                              | 段 9 の実測を見てから                                                                                                                             | 要る |

### 7.2 段 9（Chrome）と触るファイル

- 段 9 の中身は「利用者の M2 / RTX の Chrome で完走（アダプタ上限とウォッチドッグを実測してから条件を詰める）」（ADR 0118 検収表 段 9）。
  触る見込みのファイル（推測）: `tools/gpu-lab/browser/`（今は anima のタブだけで wan は無い — `ls` と `rg -l wan` で 0 件）・`examples/wan/`・
  上限次第で `packages/models/src/wan/pipeline.ts`。
- 10a・10b は新規ファイル中心で重ならない（`packages/models/src/text/`・`packages/models/src/anima/text/`〈一般化する場合〉・
  `tools/export-recipes/wan/` の新しい台本・テスト）。anima のトークナイザを動かすなら anima のレーンが範囲に入る。
- 10d は `packages/models/src/wan/pipeline.ts`・`text-embeds.ts`・`tools/export-recipes/wan/distribution.py`・`card.py`・`karume.json` を触るので、
  段 9 が pipeline に手を入れるなら重なる。段 9 の後に置くのが安全（推測）。
- 調査時点の作業ツリーには、別の作業による未コミットの変更が `tools/export-recipes/wan/`（テスト含む 18 ファイル）と exporter core
  （`emit.py`・`quant_methods.py`）・`packages/runtime/src/runtime/plan.ts` にある（`git status`）。10b / 10c が exporter core を触るなら、
  w8a8 の席の作業（`2026-10-03-wan-w8a8-recon.md`）と重なりうる。

## ADR で決めるべき軸

各軸に案と推奨。推奨は事実（上の節）と推測の両方に拠る — 推測に拠る部分は理由に書いた。

### A. トークナイザの正本と語彙外の文字

- (a) transformers 5.14.1 の実挙動（語彙外 → id 2 = `<s>`）に合わせる。今の pin の上流パイプラインと同じ id 列になる。
- (b) tokenizer.json・transformers 4.x（id 3 = `<unk>`）に合わせる。公式 Wan2.1 の当時の環境に近い。
- (c) 語彙外の文字を含むプロンプトを fail loudly で拒む。
- **推奨 (c)**。id 2 は transformers 5 の組み直しが元の T5 の語彙配置を前提にした結果で、id 3 も事前学習（byte_fallback あり）では
  出なかった入力（推測）。どちらに合わせても「上流と同じ」と言える根拠が版依存で、拒めば参照（どの版でも同じ）と TS が必ず一致する。
  受理集合は割り当て済み文字のうち 19,848 文字（CJK・仮名・Latin を含む — §1.4）。

### B. 前処理（`prompt_clean`）の移植の範囲

- (a) ftfy を全部移植 / (b) 変化しない入力だけ受理 / (b') 決定的な処理を表で移植し、mojibake 判定と entity 候補は拒否 / (c) 利用者側で正規化。
- **推奨 (b')**（§3.4）。最初の段は「`&` と 449 文字を含めば拒む」の強い版でもよい（fuzz で全件一致した範囲）。`is_bad` の移植で
  アクセント付きの欧文と `’` を受理集合に戻す。NFC は Python の UCD を正本に表を焼くか、JS の `normalize` との一致を検査する。

### C. 相対位置のバケット表の置き場

- (a) ホストが `[L,L]` の i32 を作ってグラフ入力で渡す（SBV2・Anima の入力昇格の前例）。
- (b) `[512,512]` の定数を焼いて `[:L,:L]` を切り出す（ADR 0118 決定 4 の今の文面）。
- **推奨 (a)**。前例があり、S（L）に依存する焼き込み定数をゼロに保てる。f64 と torch の f32 の一致は ±511 の全域で実測済み。
  Python / TS の生成器のバイト一致テストは前例どおり別に置く。ADR 0118 決定 4 の文面の改訂が要る。

### D. 系列長の形

- (a) 有効長 `L` の動的形（マスク入力なし）/ (b) 512 固定 + マスク入力。
- **推奨 (a)**。計算量が `L` に比例し、マスクと bias の加算が要らない。小実験では等価（§2.3・実モデルは未確認）。
  `torch.export` の 0/1 特殊化を避ける下限は anima の前例で 2（`packages/models/src/anima/text/tokenizer.ts` の `PROMPT_MIN_TOKENS`）なので、
  `L = 1`（空文字・空白だけ → `</s>` だけ）は拒む形になる。

### E. 512 トークン超の扱い

- (a) 上流どおり黙って切る / (b) fail loudly。
- **推奨 (b)**。今の recipe（`text_embeds.py:130-151`）と同じ方針で、横断の不変条件「未対応・想定外は fail loudly」に合う。

### F. 重みの格納型

- (a) i8 per-channel（ADR 0118 決定 4 の前提）/ (b) i4 g32 / (c) 両方の席。
- **推奨 (a) で始め、i4 は品質を測ってから席として足す**。mlx-umt5 の報告では group 量子化（MLX の group-affine）で出力が壊れた
  （前回調査 §6）。karume の i4 は対称の group 32 で同じ壊れ方をするかは未測（推測）。

### G. 語彙埋め込みの置き場

- (a) GPU の `embedding`（i8 で 1,002 MiB の 1 バッファ）/ (b) ホスト gather（ADR 0085 の PLE の前例）。
- **推奨: Deno は (a) で始め、段 9 の Chrome の上限の実測で (b) の要否を決める**。(a) は既存の op だけで済む。(b) は VRAM を 1 GiB 減らし
  束縛上限の問題を消すが、PLE の機構は gemma 専用の形（索引 schema・sidecar）で、Wan へ流用できるかは未確認（推測）。

### H. export のホスト RAM

- (a) 全体 1 グラフ（F32 22.7 GB を載せて trace）/ (b) fake tensor で trace して重みを逐次 emit / (c) 層を数グラフに分割。
- **推奨: 実測が先**（ADR 0118 の既定どおり — 10b）。実測で (a) が 31 GiB に入らなければ (b)。(c) はグラフと Session の本数が増え、
  text 段の admission が複雑になる（推測）。

### I. 既定の経路と事前計算の資産の扱い

- (a) GPU 経路を既定にし、事前計算の資産は退役。
- (b) 資産にあるプロンプトは資産、無いものは GPU（自動で切り替え）。
- (c) GPU 経路を既定にし、資産の経路は明示の選択で残す（umT5 を取らない軽い使い方）。
- **推奨 (c)**。(b) は同じ文字列でも資産の有無で埋め込みが変わり、黙って経路が変わる（再現性の説明がつかない）。(a) は umT5 の DL
  （i8 で約 5.3 GiB）を必須にし、今の 2.9 GiB の使い方を失う。negative の既定も同じ軸に乗る（資産の bf16 を使うか、毎回 GPU で作るか —
  毎回でも 126 トークンで 0.59 T MAC と軽い〈推測〉）。

### J. 数値の門

- 移植の門 = fake-quant 重みの層逐次参照に対する帯（実測導出・受入れは別ケース）。品質 = bf16 資産・f32 参照との 3 点比較の記録 + 動画の目視。
  sha 行は新しい case id で足す（§6）。
- **推奨: この二層**。「GPU i8 vs bf16 資産」を門にすると、bf16 自身の誤差が床になり帯を決める根拠が無い（推測）。

## 出典

- 実物（HF キャッシュ・pin した revision `0fad780a534b6463e45facd96134c9f345acfa5b`）:
  `~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a…/text_encoder/config.json`・
  `model.safetensors.index.json`・各 safetensors のヘッダ・`tokenizer/{spiece.model,tokenizer.json,tokenizer_config.json,special_tokens_map.json}`・
  `model_index.json`
- transformers 5.14.1（`tools/.venv/lib/python3.14/site-packages/transformers/`）: `models/umt5/modeling_umt5.py`・`models/t5/tokenization_t5.py`・
  `activations.py`
- diffusers 0.39.0: `pipelines/wan/pipeline_wan.py`
- ftfy 6.3.1: `ftfy/__init__.py`・`fixes.py`・`badness.py`・`chardata.py`
- 公式 Wan2.1（commit `9737cba9c1c3c4d04b33fcad41c111989865d315`）:
  <https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/tokenizers.py>・
  <https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/requirements.txt>
- google/umt5-xxl（sha `66cb9e7e85526fe440a945569e42c72fb6cbc0ad`）: <https://huggingface.co/api/models/google/umt5-xxl?blobs=true>・
  <https://huggingface.co/google/umt5-xxl/resolve/66cb9e7e85526fe440a945569e42c72fb6cbc0ad/tokenizer.json>
- transformers 4.57.1 の挙動: `uv run --no-project --with 'transformers==4.57.1' --with tokenizers --python 3.12`（scratchpad で実行・リポの venv に触れない）
- リポ: `tools/export-recipes/wan/{sources.py,text_embeds.py,prompts.py,pipeline_ref.py}`・`packages/models/src/wan/{text-embeds.ts,pipeline.ts}`・
  `packages/models/src/anima/text/{t5-tokenizer.ts,spm-normalizer.ts,tokenizer.ts}`・`packages/models/src/text/unigram.ts`・
  `packages/models/tests/{anima_tokenizer_test.ts,sbv2_rel_pos_parity_test.ts,e2e_anima_test.ts}`・`packages/runtime/src/ops/{names.ts,shapes.ts}`・
  `tools/exporter/src/karume/{normalize.py,aten_handlers.py,convert.py,quantize.py}`・`tools/export-recipes/anima/{text.py,distribution.py}`
- docs: ADR 0118・0026・0034・0085・0090・0109・`docs/quantization.md`・`docs/research/2026-10-02-video-gen-recon.md`・
  `docs/research/2026-10-02-b570-vram-budget.md`・`docs/research/2026-10-03-wan-w8a8-recon.md`・`docs/backlog.md`
