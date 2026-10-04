> 時点スナップショット（2026-10-04）— 依存モデルの差し替え（部品差し替え席 `components`）の現状・穴・設計候補の読み取り調査。正本ではない。

# 依存モデルの差し替え — 部品差し替え席の現状と穴（Wan の umT5 を例に）

読み取りだけの調査（HEAD `cc637c2f`・GPU 不使用・リポのコード変更なし）。CPU の合成実験は 1 本だけ回した
（tiny umT5 を `/tmp` へ export — §1.2.3）。きっかけは「Wan の umT5 を追加学習版（コミュニティが公開している互換モデル）へ
差し替えたい」という利用例で、同じ需要は他の系列でも続く見込み。umT5 の重みの中身（キー名・dtype・トークナイザの同一性）は
別レッグの担当で、ここでは扱わない。

事実（§1）と推測（§2）を節で分ける。事実には `file:line`・実行したコマンドと観測値を付ける。一次ソース = リポのコード・
ADR・手元の配布形ミラー（`models/`）・自分で回した実験。二次ソース = WebFetch の要約経由で読んだ外部記事。

## 要約

1. **差し替えの仕組みは既にある**。8 系列の `fromPretrained` は `components: { <役割>: { source, model?, quant? } }` を
   受け、`packages/models/src/hub/components.ts:381-492` の 1 本が処理する。受理の条件は「グラフ記述の sha256 が元の
   manifest の宣言と同一」（`components.ts:432-437`）＋束縛表の不足 / 余剰 0（`openContainer`）＋家族の門。Wan では
   `text_encoder` を差し替えられる（`textEncoder: "gpu"` の経路のとき — `wan/pipeline.ts:188-191`）。
2. **追加学習版を同じ recipe で export すれば、グラフ記述はバイト同一になる見込みが高い**。グラフ記述は形・op・attr・
   重み非依存の定数だけを持ち、重み・scale・行の塊の区切り・`fixed_weights` は全部モデル記述側にある。umT5 の書き手は
   そもそも重みを持たない meta モデルで trace する（`wan/umt5_export.py:22-24`）。合成実験でも「重みだけ違う 3 本の
   グラフ記述 sha256 が一致・語彙 +1 行で不一致」を観測した（§1.2.3）。実ミラーでも anima の 7 モデル（公式 5 + 第三者
   2）の `transformer` が同じグラフ sha256 を持つ（§1.2.4）。
3. **トークナイザは元の manifest（Wan）の資産のまま**で、差し替え先からは来ない（`components.ts:486-489`）。トークナイザの
   同一性は誰も検査しない。語彙数が同じでトークナイザが違うモデルを差すと、受理されたまま誤った id 列で回る。
4. **格納型が違う差し替え**（例 i4）はグラフ記述では通る（格納はモデル記述側 — 実ミラーで f16 / i8 / i4 が同じグラフ
   sha256）が、Wan は家族の門 `umt5Contract` が i8 / f32 以外の格納を重みを取る前に拒む（`wan/pipeline.ts:994-1008`）。
   差し替え先の quant 席の `session` / `gpuFeatures` / `requiredLimits` は読まれず、元の席の宣言が使われる。
5. **主な穴は 3 つ**: ①第三者が互換部品を作る経路が無い（umT5 の recipe と配布の門が Wan の pin した上流 1 本に固定 —
   `wan/umt5_distribution.py:156-158, 278-290`）②発見性と説明が無い（README / モデルカード / examples に差し替えの記述が
   0 件・テストは汎用 4 本 + irodori の拒否経路 1 本だけ）③付随資産（トークナイザ）が役割に結び付いていない。
6. **推奨は案 A**（現状の席を保ち、第三者の export 経路・カードの互換表示・Wan の正の経路のテスト・文書を足す）。
   付随資産の結び付け（案 B の一部）は「同じ語彙数で違うトークナイザ」の実需が出た時点の裁定に回す（§3）。

## 1. 事実

### 1.1 今のコードでの書き方とテスト

#### 1.1.1 呼び出し例（Wan の umT5 を別リポの `text_encoder` へ）

公開の型は `FromPretrainedComponentOptions.components`（`packages/models/src/hub/load-options.ts:59-70`）と
`ComponentSource = { source: string | HubRepoRef | DistributionSource; model?; quant? }`
（`packages/models/src/hub/components.ts:120-124`）。Wan は `fromPretrained` で `options.components` を
`loadContainerComponents` へ渡す（`packages/models/src/wan/pipeline.ts:1197-1201`）。

```ts
import { WanPipeline } from "@karume/models/wan";

// HF（差し替え先は karume/5 の配布リポで、manifest の weights に text_encoder を持つこと）
const wan = await WanPipeline.fromPretrained(
  { repo: "hdae/karume-wan2.1", revision: "<40 桁>" },
  {
    textEncoder: "gpu", // 既定。"precomputed" では text_encoder を開かないので差し替えも効かない
    components: {
      text_encoder: {
        source: { repo: "<owner>/<互換 umT5 の karume リポ>", revision: "<40 桁>" },
        model: "xxl", // 省略時は差し替え先の defaultModel
        quant: "i8", // 省略時は差し替え先の defaultQuant
      },
    },
  },
);
```

手元の配布形なら `source` に取得元ハンドル（`@karume/hub/deno` の `denoDirectory("<dir>")`）を渡す
（`toManifestSource` が `DistributionSource` をそのまま受ける — `packages/models/src/hub/repo-ref.ts:89-94`）。差し替えた後は
元の席の part は 1 バイトも取られない（`container_components_test.ts:115-118` が汎用の形で確認）。ローカル取得元が越境先を
解くのは越境の FileRef を開くときだけ（`packages/hub/src/sources/local.ts:207-213` の `originFor`）なので、`text_encoder` を
差し替えるなら元の Wan の取得元に umT5 の `crossRepo` mapping は要らない見込み（推測 — 越境の席を差し替える形のテストは無い）。

`fromAssets`（取得済みバイト列の入口）には `components` の席が無い。呼び手が `assets["text_encoder"]` に任意の容器を
入れれば、期待値（2 文書の sha256）の照合なしで家族の門だけを通る（`components.ts:27-33, 272-289`）。

#### 1.1.2 処理の順序（`loadContainerComponents`）

| 段 | 何をするか                                                                                                            | 根拠                                             |
| -- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 0  | `components` の役割が `componentKeys` の部分集合か（未知は列挙して落とす）                                            | `components.ts:390-397`                          |
| 1  | 差し替え先の manifest を `loadManifest` → `resolveSelection(…, { model, quant, weights: [役割] })`                    | `components.ts:416-430`                          |
| 2  | 差し替え先の `descriptor.graph.sha256` と元の宣言を比べる（2 つの manifest だけで判定・容器は 1 本も取らない）        | `components.ts:431-437`                          |
| 3  | 各部品の part 0 だけを温めて `openContainer`（2 文書の期待値照合・束縛の不足 / 余剰・codec 台帳）→ `prepareContainer` | `components.ts:461-471`                          |
| 4  | 家族 admission（Wan は `#admit` — `umt5Contract` を含む）                                                             | `components.ts:476`・`wan/pipeline.ts:1244-1342` |
| 5  | 重みの part を取得元ごとにまとめて温める                                                                              | `components.ts:480-484`                          |
| 6  | 元の manifest の `assets`（トークナイザ等）を全量で取る                                                               | `components.ts:486-489`                          |

#### 1.1.3 テストの有無

| テスト                                         | 見ていること                               | 場所                                                        |
| ---------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------- |
| 差した役割だけが別リポから来る                 | 正の経路（汎用・疑似 HF 2 リポ・GPU なし） | `packages/models/tests/container_components_test.ts:94-123` |
| グラフ記述が違えば重みを 1 本も取らずに落ちる  | 拒否の経路                                 | 同 `:125-145`                                               |
| 未知の役割名は落ちる                           | 綴り違い                                   | 同 `:147-166`                                               |
| 同じ path が 2 リポに並んでも進捗は 2 本       | 進捗の集約                                 | 同 `:168-202`                                               |
| irodori の `dit` を差し、次元が違えば落ちる    | 家族の配線（拒否の経路だけ）               | `packages/models/tests/irodori_admission_test.ts:602-667`   |
| barrel とサブパスで `ComponentSource` が同じ型 | 型面                                       | `packages/models/tests/models_barrel_surface_test.ts:28-34` |

無いもの（`rg -n "components:" packages/*/tests examples tools/published-smoke` の結果から）: Wan を含む家族単位の
**正の経路**、越境参照の席（Wan の `text_encoder` は元から越境参照）を差し替える形、差し替え先が別の格納型・別の
quant 席を持つ形、`fromAssets` で別容器を入れる形。

### 1.2 受理の条件 — 「グラフ記述の sha256 が同一」の中身

#### 1.2.1 グラフ記述に入るもの / 入らないもの

仕様（`docs/container-v1.md:88-168, 170-257, 271-283`）と umT5 の実物の descriptor（`models/karume-umt5-xxl/xxl/text_encoder/
model.i8-00001-of-00026.krm` の part 0 を node で切り出し — 付録 A）から:

| 項目                                                                                                                                                   | 置き場                       | 実物（umT5 i8）                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `format` / `version` / `capabilities.ops`（全グラフの `requires.ops` の和集合）                                                                        | グラフ記述                   | ops 11 種（add / bmm / embedding / expand / gelu_tanh / linear / mul / permute / reshape / rms_norm / softmax） |
| IR v2 本体: `symbols` / `inputs` / `outputs` / `initializers`（名前だけ — 欄は空） / `values`（全中間値の dtype と形） / `nodes`（op・入出力名・attr） | グラフ記述                   | ノード 914・initializer 244・記号 `L`・attr の浮動小数は `rms_norm` の `eps: 0.000001` だけ                     |
| 形（語彙埋め込み `[256384, 4096]`・相対位置の表 `[32, 64]` など）                                                                                      | グラフ記述（`values`）       | 上の値                                                                                                          |
| const 領域の目次と束縛（`const.blocks[].sha256`・`const.constants[]`）                                                                                 | グラフ記述                   | 2 本: f32 `[10240]` と `[4096]`（linear の bias 席の 0 埋め — part 1 の 57,344 B は全バイト 0）                 |
| 量子化の scale・codec・`encoding`                                                                                                                      | モデル記述（束縛表）         | `codecs: ["f32", "int8-sym"]`                                                                                   |
| 行の塊の区切り（piece 列）                                                                                                                             | モデル記述（束縛表）         | `encoder.embed_tokens.weight` は `pieces: [{block, rows: [0, 8192]}, …]`                                        |
| `fixed_weights`（固定 packed 値の入口）                                                                                                                | モデル記述（束縛表と block） | 書き手は `fixed_weights` で渡す（`wan/umt5_export.py:25-29`）・グラフには現れない                               |
| part / block の配置・`provenance`                                                                                                                      | モデル記述                   | `provenance.upstreamRevision = 0fad780a…`                                                                       |

const の名前は値の内容ハッシュ（`const.<sha256 の先頭 16 hex>` — `tools/exporter/src/karume/convert.py:638-680`）。
const に入るのは「定数と shape 記号だけに依存する部分木」で、**パラメータ / バッファ経由の畳み込みは対象外**
（`convert.py:552-557`）。グラフ記述の直列化は正準 JSON（`tools/exporter/src/karume/container.py:200-264` — ノードの
`attrs` は code point 順に並べ替え、それ以外は組み立て側の固定順）。

#### 1.2.2 umT5 の書き手が重みに依存しない理由

`wan/umt5_export.py:22-24`: 「trace は重みを持たない（meta）上流のモデルで回す…グラフは実重みの export と JSON で同一に
なる（pytest が小模型で縛る）」。meta モデルは `UMT5Config.from_pretrained(directory)` だけから組む
（`umt5_export.py:357-364`）。つまりグラフ記述を決める入力は `config.json` ＋ラッパ（`wan/umt5_patch.py`）＋ exporter /
torch / transformers の版。

#### 1.2.3 合成実験（CPU・tiny umT5）

`tools/export-recipes/wan/umt5_probe.py` の `TINY_CONFIG`（vocab 384・d_model 64・4 層）を使い、`export_probe` →
`publish_model`（pytest `wan/tests/test_umt5_patch.py:205-226` と同じ引数）で `/tmp` に容器を書き、part 0 の 2 文書の
sha256 を比べた（スクリプトは付録 B）。

| 変種           | 中身                                                   | グラフ記述 sha256         | モデル記述 sha256       |
| -------------- | ------------------------------------------------------ | ------------------------- | ----------------------- |
| base_i8        | seed 20261003                                          | `ff5d6e96…2652ef`         | `4350005c…72a1`         |
| base_i8_again  | 同じ重みで 2 回目                                      | `ff5d6e96…2652ef`（同）   | `4350005c…72a1`（同）   |
| tuned_i8       | base に全パラメータ ±1e-2 相対の摂動（追加学習の模擬） | `ff5d6e96…2652ef`（同）   | `1101f1dc…0b42`（違う） |
| other_seed_i8  | 別 seed（重みが全部違う）                              | `ff5d6e96…2652ef`（同）   | `7b8cf0bb…23bc`（違う） |
| base_f32       | base を f32 格納で                                     | `ff5d6e96…2652ef`（同）   | —                       |
| vocab_plus1_i8 | `vocab_size` を +1                                     | `082dbb23…75cc68`（違う） | 違う                    |

f16 格納は「重みが f16 で表せない値を含む」で emit が拒否（前もって f16 へ丸める工程が要る — 想定どおりの拒否で、
グラフの比較はしていない）。

#### 1.2.4 実ミラー横断の観測

`models/*/karume.json` の全 (モデル, 役割, dtype) の `descriptor.graph.sha256` を jq で列挙した（付録 A）:

- **anima**: 公式 5 モデル（`karume-anima`）と第三者の追加学習 2 モデル（`karume-anima-extra` の `anima-wai-v1.0` /
  `anima-copycat-20260610`）の `transformer` が全部 `c605022b5b6f…`。f16 と i8 も同じ。
- **sbv2**: 話者 4 本（F1 / F2 / M1 / M2）の `front` / `voice` / `text_encoder` がそれぞれ同じ sha256（f16 / i8 / i4 も同じ）。
- **irodori**: v4-small と v4.1-small（重みの版違い）で 9 役割すべて同じ。f32 / i8 / i4 も同じ。
- **例外**: irodori の `backbone` だけ f16 席（`7774443f…`）が f32 / i8 席（`91d7412c…`）と違う。ノード 957・initializer 160・
  const 長 1,609,728 B は同じで、違うのは形 `[1,1,512,64]` の const 4 本の内容（＝内容ハッシュの名前）。f16 の書き手は
  f32 のパラメータとバッファを f16 へ丸める（`tools/export-recipes/irodori/export.py:2572-2597`）。
- **Wan の umT5**: `karume-wan2.1` の `text_encoder` と `karume-umt5-xxl` の `text_encoder` は同じ `59ca0ccf…`（越境参照
  なので当然）。

#### 1.2.5 グラフ記述が同一にならない要因（列挙）

| 要因                                                                          | 根拠                                                                                            | umT5 の追加学習版で起きるか                                                  |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 構成（`config.json`）の違い — 語彙数・層数・幅・head 数・`eps`・活性の種類    | §1.2.3 の vocab+1・§1.2.2                                                                       | 語彙を足した派生なら起きる。普通の追加学習なら起きない                       |
| 書き出しの工程が const の値を変える（バッファの丸め等）                       | §1.2.4 の irodori backbone f16                                                                  | umT5 の const は 0 埋めの bias 2 本だけなので起きない                        |
| グラフへ重み由来の数を焼く方式（QAT の活性 scale・`static_quantize` の attr） | ADR 0108 決定 1・追記 5（`0108-container-format.md:101-102, 693-697`）                          | umT5 の i8 は重みだけの量子化で、活性は Session のノブ（動的）なので起きない |
| exporter / torch / transformers の版で trace の結果が変わる                   | ADR 0108 決定 4（`0108-container-format.md:171-173`）が「生成器の版でバイト違い」の経路を名指し | 推測 §2.2                                                                    |
| ラッパ（`umt5_patch`）・名前の付け方の変更                                    | IR の initializer 名 = モジュールの FQN                                                         | 同じ recipe なら起きない                                                     |
| 格納型の違い                                                                  | 起きない（§1.2.3 の f32 / i8・§1.2.4 の f16 / i8 / i4）                                         | —                                                                            |

### 1.3 付随資産（トークナイザ）

- 差し替えで取るのは**容器だけ**。資産は常に元の manifest の `selection.assets` から取る（`components.ts:486-489`）。
  差し替え先の manifest の `assets` は読まれない（`resolveSelection` の戻りのうち `containers[key]` だけを使う —
  `components.ts:422-427`）。
- Wan のトークナイザ資産 `umt5_tokenizer` は Wan リポの自前の資産（ADR 0119 追記 C — `0119-wan-umt5-gpu-text-encoder.md:483-484`）。
  形式 `karume-wan-umt5-tokenizer/1` は Wan の前処理の表と束ねたもの。`karume-umt5-xxl` は資産を持たない
  （`models/karume-umt5-xxl/karume.json` の `assets: {}`）。
- 構築時の門はトークナイザの `maxLength` が 512 か・既定の negative が通るかだけ（`wan/pipeline.ts:1366-1390`）。
  トークナイザと text_encoder の対応（同じ語彙か）を検査するものは無い。
- したがって: **語彙数が違う**互換モデルはグラフ記述が変わるので差し替えで拒否される（§1.2.3）。**語彙数が同じでトークナイザが
  違う**モデルは受理され、Wan のトークナイザの id 列で回る（沈黙の誤り）。トークナイザが違う互換モデルを正しく扱う手段は無い。
- anima-extra は資産（`tokenizer` / `tokenizer_2`）も `hdae/karume-anima` へ越境参照している（`models/karume-anima-extra/karume.json`）
  — 越境参照は資産にも効くが、差し替え席は資産を持たない。

### 1.4 quant 席との関係

- **グラフ記述は格納型に依らない**（ADR 0108 決定 17「1 アーキ・1 量子化方式・1 グラフ」— `0108-container-format.md:391-397`・
  実測 §1.2.3 / §1.2.4）。差し替え先が別の格納型でも①グラフ記述の照合は通る。
- ②`openContainer` が codec 台帳の突合（未知の codec は重み取得前に拒否 — container-v1 §2.2 の `codecs`）。
- ③家族の門: Wan の `umt5Contract` は格納を i8（`int8-sym`）と f32 だけ受け、i8 が 1 本も無い容器も拒む
  （`wan/pipeline.ts:986-1008`・ADR 0119 決定 5）。**i4 / f16 の umT5 は重みを取る前に落ちる**。sbv2 の `text_encoder`
  （DeBERTa）は配布形自体が i4 / i8 を持つ（`models/karume-sbv2-jvnv/karume.json`）ので、家族ごとに受ける格納は違う。
- 差し替え先の quant 席の `session` / `gpuFeatures` / `requiredLimits` は**使われない**。Wan は元の席の値で
  `resolveSessionOptions` / `assertRequiredLimitsBeforeDownload` を回す（`wan/pipeline.ts:1186-1196, 1293-1316`）。umT5 の
  Session は `{}` で張る（`wan/pipeline.ts:1538`）ので、Wan では席のノブが umT5 に効くことはない。
- ADR 0109 決定 10 ③「席の実行ノブと codec の整合」（`0109-manifest-v5-container.md:169-175`）に当たる専用の検査は
  `components.ts` に無い（家族の門と runtime の capability 門に任されている）。

### 1.5 全系列の棚卸し — 内部で使う依存モデル

部品キーは `loadContainerComponents` に渡す配列（各 `pipeline.ts`）、出所は各 README の `base_model` と
`karume.json`、越境は manifest の FileRef の `repo` の有無から。

| 系列（pipeline）       | 差し替えられる役割（componentKeys）                                                                                                          | 内部の依存モデル                                                                      | 同梱 / 越境                                                                                        | 付随資産                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| wan/1                  | transformer / vae_decoder_first / vae_decoder_next / text_encoder（gpu 経路のみ — `wan/pipeline.ts:188-191`）                                | umT5-XXL encoder（text_encoder）・Wan VAE decoder                                     | text_encoder は `hdae/karume-umt5-xxl` へ越境（revision は仮の 40 桁 0）・他は同梱                 | text_embeds・umt5_tokenizer（自前）                             |
| anima/1                | text_encoder / text_conditioner / transformer / vae_decoder（`anima/pipeline.ts:152`）                                                       | Qwen3（text_encoder — `anima/pipeline.ts:6-7`）・LLM adapter（text_conditioner）・VAE | 公式リポは同梱。anima-extra は text_encoder と vae_decoder を `hdae/karume-anima@adb9dcf0…` へ越境 | tokenizer（Qwen2 BPE）・tokenizer_2（T5 Unigram）— extra は越境 |
| sbv2/1                 | front / voice / text_encoder（`sbv2/pipeline.ts:131`）                                                                                       | DeBERTa-v2-large-japanese-char-wwm（text_encoder — README の base_model）             | 同梱（話者 4 本で同じ text_encoder）                                                               | tokenizer・symbols・style_vectors・speaker_embeddings           |
| irodori/1              | backbone / text_proj / caption_proj / speaker / duration / dit / dit_context / codec_decoder / codec_encoder（`irodori/admission.ts:51-61`） | ModernBERT-ja-310m 由来の backbone・Semantic-DACVAE（codec_*）                        | 同梱                                                                                               | tokenizer                                                       |
| gemma4/1・gemma4-qat/1 | model（+ drafter — 投機時のみ・`gemma/pipeline.ts:942`）                                                                                     | drafter（gemma-4-E2B-it-assistant）                                                   | 同梱                                                                                               | tokenizer                                                       |
| siglip2/1              | vision（`siglip2/pipeline.ts:104`）                                                                                                          | —                                                                                     | 同梱                                                                                               | —                                                               |
| birefnet/1             | matte（`birefnet/pipeline.ts:123`）                                                                                                          | —                                                                                     | 同梱（lucida は別リポの派生）                                                                      | —                                                               |
| depth-anything/1       | depth（`depth-anything/pipeline.ts:148`）                                                                                                    | —                                                                                     | 同梱                                                                                               | —                                                               |
| vowel-detector/1       | crnn（`vowel-detector/pipeline.ts:140`）                                                                                                     | —                                                                                     | —                                                                                                  | —                                                               |
| umt5-encoder/1         | （読む TS の家族は無い — 役を名乗るだけ・`wan/umt5_distribution.py:59-63`）                                                                  | —                                                                                     | —                                                                                                  | —                                                               |

「依存モデルを他の系列と共有している」のは今のところ Wan → umT5 と anima-extra → anima の 2 本だけ。

### 1.6 穴の列挙

1. **発見性**: 互換な部品を探す手段が無い。カードには越境先は出る（`models/karume-wan2.1/README.md:277`）が、
   「この役割はグラフ記述 `59ca0ccf…` と互換なら差し替えられる」という表示も、互換リポの一覧も無い
   （`rg -n -i "sha256|compatib|swap|replace|components" models/*/README.md` は FileRef の説明だけ）。
2. **書き味**: 役割名・model 名・quant 名を利用者が知っている前提。差し替え先の pipeline 名
   （`umt5-encoder/1` 等）は読まれない（`components.ts:418-427` は manifest を読むが pipeline を見ない）。
3. **互換性の宣言**: 役割の契約を名乗る識別子は無く、グラフ記述の sha256（内容ハッシュ）が事実上の識別子。`umt5-encoder/1`
   は「役を名乗る」と書かれているが解釈する実装は無い（`wan/umt5_distribution.py:59-63`）。
4. **第三者が互換部品を作って配る手順が無い**:
   - umT5 の書き手は上流を `wan.sources.SOURCES` の pin 1 本からしか読まない（`umt5_export.py:766` の `--model` は
     `choices=sorted(SOURCES)`・`wan/sources.py:54-63` は `t2v-1.3b` だけ）。
   - 配布の門 `assert_umt5_encoder` は容器の出所を `UMT5_UPSTREAM` の pin と突き合わせる（`umt5_distribution.py:156-158,
     278-290` — 「束縛表と入出力の形は同じ構造の別の checkpoint でも通るので、出所でしか閉じられない」）。別の checkpoint
     から焼いた umT5 は `dist.py --pipeline umt5` を通らない。
   - 配布側で差し替えた Wan の配布形（anima-extra 型）を作る `dist --ref-*` は「参照先のバイト列が自分で組むはずのバイト列と
     一致すること」を要求する（`tools/exporter/src/karume/dist.py:1790-1797`）ので、Wan の計画自体の umT5 を差し替えない限り
     越境先だけ付け替えた配布形は作れない。Wan の計画も `assert_umt5_encoder` を通る（`wan/distribution.py:663`）。
   - recipe は wheel の外（`tools/export-recipes/`・ADR 0065）なので、第三者はリポを clone して recipe を足すしかない。
5. **不一致時のエラー文言**: グラフ記述の不一致は 2 つの sha256 を並べるだけ（`components.ts:432-437`）。どこが違うか
   （語彙数・層数など）は出ない。宣言だけで落とす設計のため descriptor をまだ取っていない。
6. **pin と上書き**: Wan の manifest は umT5 を 40 桁 revision で pin する（ADR 0119 追記 A）。`components` に文字列の
   `source` を渡すと `{ repo }` = main 追従になり（`wan/pipeline.ts:1145-1146` と同じ規約）、元の pin は消える。上書きした事実は
   どこにも記録されない（再現性は呼び手の責任）。
7. **付随資産の結び付けが無い**（§1.3）: トークナイザが役割と結び付いておらず、語彙数が同じ別トークナイザは沈黙で通る。
8. **差し替え先の quant 席の宣言が無視される**（§1.4）: 元の席の `requiredLimits` で事前判定するので、差し替え先が大きい
   バッファを要する形のとき、事前判定を通って Session 構築で落ちうる（推測 §2.3）。
9. **README / docs**: 用語集の 1 行（`docs/glossary.md:78`）と ADR・JSDoc だけ。README・examples・モデルカードに記述 0 件
   （`rg -n components README.md models/README.md examples/*/README.md` が 0 件）。
10. **テスト**: §1.1.3 の「無いもの」。

### 1.7 外部の利用例（二次ソース）

紹介記事（WebFetch の要約経由で読んだ二次ソース — URL とリポ名は git 追跡外の記録
`.claude/reviews/2026-10-04_umt5-variant-recon/` に置く）の要約: 差し替え先は HF 上の追加学習版で、fp8 版と bf16 版がある、ComfyUI で使う、bf16 版にはトークナイザが入っておらず fp8 版の中の
トークナイザを移植する必要がある、検証は Wan2.2-Remix が中心で Wan2.1 への適用は明記なし。重みとトークナイザの中身は
別レッグの担当。

## 2. 推測（事実と分ける）

### 2.1 追加学習版 umT5 の export はグラフ記述が一致する見込み

- 根拠: §1.2.2（meta trace）・§1.2.3（重みだけ違う 3 本が一致）・§1.2.4（anima の第三者 2 本が一致）。
- 前提: 追加学習版の `config.json` の構造の欄が Wan の pin と同じ（語彙 256,384・d_model 4,096・d_ff 10,240・24 層・
  64 head・d_kv 64・バケット 32 — 実物の形から逆算）。語彙を足していればグラフは変わる。ComfyUI 形式の checkpoint には
  HF の `config.json` が付かない可能性があり、その場合は Wan の pin の `config.json` を流用する形になる（未確認）。
- 同じ理由で、**google/umt5-xxl から焼いた umT5 も、encoder の構成が同じなら Wan の宣言と同じグラフ記述になり、差し替え席で
  受理される**見込み（umt5-xxl の config の突合は未実施）。

### 2.2 版の違いで trace の結果が変わる

exporter / torch / transformers の版が違うと分解の形が変わり、グラフ記述が別になりうる。ADR 0108 決定 4 が経路を名指し
しているが、版を跨いだ実測は無い。第三者が別の版で export した互換部品は拒否される可能性がある（安全側の失敗）。

### 2.3 差し替え先の requiredLimits が無視される影響

Wan の umT5 は i8 / f32 しか受けないので、今の Wan では影響は出ない見込み。格納型の幅が広い家族（sbv2 の text_encoder 等）で
差し替え先がより大きいバッファを要すると、元の席の `requiredLimits` の事前判定を通って Session 構築で落ちる形がありうる
（実例は未確認）。

## 3. 設計候補

前提の事実: 構造の互換（グラフ記述の同一）は内容ハッシュで既に機械的に保証されている（§1.2）。足りないのは「作る経路」
「見つける手段」「付随資産」「説明」（§1.6）。

### 案 A — 今の席のまま、作る経路・表示・テスト・文書を足す

- 作る経路: umT5 の書き手と配布の門に「pin 済みの任意の上流」を足せる口（`wan.sources` / `UMT5_UPSTREAM` の表への追加、
  または checkpoint の path と出所を引数で名乗る口）。出所の門（`assert_upstream_provenance`）は外さず、表に載った出所ごとに
  掛ける。
- 表示: モデルカードに「差し替え互換」節（役割名・グラフ記述の sha256・差し替えの書き方の例）。互換部品側のカードには
  「どの系列のどの役割に差せるか」。HF の tags でも探せるようにする。
- テスト: Wan の正の経路（疑似 HF 2 リポで text_encoder を差す・元の越境先を 1 本も取らない）と、i4 の差し替えが
  `umt5Contract` で重みを取る前に落ちる形。
- 文書: `docs/` に手順の頁、README に節、不一致の文言に「グラフ記述が違う主な理由（構成・語彙・版）」の案内。
- 得: manifest 形式は変えない（`karume/5` のまま）。umT5 の追加学習版のように構成とトークナイザが同じ部品はこれで足りる。
- 失: §1.6 の 7（同じ語彙数の別トークナイザが沈黙で通る）は残る。発見性はカードと tags 頼み。

### 案 B — 案 A ＋ 役割の契約と付随資産を manifest に名乗らせる

- 部品側（例 umT5 リポ）の manifest が「提供する役割の契約」（例 `umt5-xxl-encoder/1`）と、その役割が要る付随資産
  （トークナイザ）を持つ。元側は「この役割は契約 X を要求し、資産 Y はこの役割に属する」と宣言する。差し替えると、役割に
  属する資産も差し替え先から取るか、同一（sha256）を検査する。
- 得: トークナイザの沈黙の誤りが消える。契約名で互換リポを検索できる。
- 失: manifest の major を上げる（ADR 0038 決定 1 の未知キー拒否 — 0038 §7 の列挙に無い追加は major）。契約名はグラフ
  記述の sha256 から導けない別の事実で、両方を持つと食い違う状態が生まれる（契約名は同じでグラフが違う、の扱いが要る）。
  Wan のトークナイザ資産は Wan の前処理の表と束ねた形式なので、資産を部品側へ移すには形式を割る作業が要る（ADR 0119 追記 C）。

### 案 C — 依存モデルを「部品リポ」として一級化し、差し替え = 越境先の付け替えにする

- 依存モデル（容器 + 付随資産）を独立の配布リポにし、元の manifest は越境参照だけを持つ（Wan → umT5 の今の形を一般化し、
  トークナイザも部品リポ側へ）。利用者の差し替えは「越境先リポの付け替え」（package manager の overrides に近い）で表す。
- 得: 付随資産ごと差し替わる。第三者は部品リポを 1 本作れば済み、元の系列の再配布は要らない。
- 失: 全系列の配布形の作り直しと hub / models の両方の変更。依存モデルを他系列と共有しているのは今 2 本だけ（§1.5）で、
  投資に対して適用先が少ない。

### 推奨 — 案 A（付随資産の結び付けは実需が出た時点で案 B の一部として裁定）

決め手:

1. 目の前の利用例（umT5 の追加学習版）は構成とトークナイザが同じ見込みで（§2.1）、今の席の受理条件で通る。止めているのは
   「作る経路」（§1.6 の 4）と説明の欠落で、manifest の変更は要らない。
2. 契約名を足すと、グラフ記述の sha256 から導けない第 2 の識別子を持つことになり、食い違う状態が生まれる（導出できる状態を
   別欄で持たない規律）。互換の判定は内容ハッシュのままがよい。発見性はカードと tags で足りる。
3. トークナイザの沈黙の誤り（§1.6 の 7）は残るが、発火条件は「語彙数が同じで別のトークナイザ」に限られる。案 A では
   limitations に記録し、最初の実例が出た時点で「役割に属する資産」の宣言（案 B の後半だけ）を裁定する。

## 4. 未確定

- 追加学習版 umT5 の構成・キー名・トークナイザが Wan の pin と同じか（別レッグ）。
- google/umt5-xxl の encoder の `config.json` が Wan の pin とグラフに効く欄で同じか（§2.1 — 未突合）。
- exporter / torch / transformers の版を跨いだグラフ記述の同一性（§2.2 — 未実測）。
- `karume-umt5-xxl` の公開と実 SHA（今の Wan の越境 revision は仮の 40 桁 0 — `models/karume-wan2.1/karume.json`）。HF 上での
  差し替えの通しはそれまで試せない。

## 付録 A — 実行したコマンド（抜粋）

```sh
# 全配布形の (モデル, 役割, dtype) → グラフ記述 sha256 と越境先
for f in models/*/karume.json; do jq -r '.models | to_entries[] | .key as $m | .value.weights | to_entries[]
  | .key as $r | .value | to_entries[] | "\($m)\t\($r)\t\(.key)\tgraph=\(.value.container.descriptor.graph.sha256[0:12])
  \tcross=\(.value.container.parts[0].repo // "-")"' $f; done

# part 0 から 2 文書を切り出す（ヘッダ: magic 4 B・u32・u64 グラフ長・u64 モデル長）
node dump.mjs models/karume-umt5-xxl/xxl/text_encoder/model.i8-00001-of-00026.krm umt5
#   → KRMC graphLen 166102 modelLen 137037・graphs [text_encoder]・const 57344 B / 2 本・nodes 914・initializers 244
#   → codecs [f32, int8-sym]・provenance.upstreamRevision 0fad780a534b6463e45facd96134c9f345acfa5b
# const の part（part 2 = model.i8-00002-of-00026.krm 57,344 B）の非 0 バイト数 → 0
```

## 付録 B — 合成実験のスクリプト（`/tmp/claude-1000/recon-2026-10-04/component-swap/exp.py`）

```sh
cd tools/export-recipes && PYTHONPATH=$PWD uv run --group wan --inexact python \
  /tmp/claude-1000/recon-2026-10-04/component-swap/exp.py /tmp/claude-1000/recon-2026-10-04/component-swap/exp
```

中身: `umt5_probe.tiny_model(SEED)` を base とし、①同じ重みで 2 回②全パラメータに `randn × 1e-2 × mean|p|` の摂動
③`tiny_model(SEED + 1)`④`vocab_size + 1` の `UMT5EncoderModel` をそれぞれ `export_probe` → `publish_model(graph_name=
"text_encoder", weight_dtype="i8", weight_scales, weight_dtype_overrides)` で書き、part 0 の 2 文書の sha256 を出す。
f32 / f16 格納は `exp2.py` で同じ base を `weight_dtype` だけ変えて書いた。結果は §1.2.3 の表。

## 独立検証（2026-10-04）

別のレッグが、設計が依存する主張 14 個を一次ソースと再計測で確かめた。成り立つ（holds）が 13 個。

### 覆った・不確かな主張

- **refuted（影響 medium）**: （§1.2.5）config の構造の欄が違えばグラフ記述は変わる（列挙が網羅的という前提） — tiny で relative_attention_max_distance を 128→64 にすると、グラフ sha もモデル sha も base と完全一致した（容器がバイト同一）。バケット表はグラフの入力で、TS 側が maxDistance 128 を固定している（wan/umt5/relative-position.ts:39）

### 本文が落としていた、設計に効く事実

- 互換の契約にはグラフの外の前処理定数も入る: バケット表の numBuckets/maxDistance（relative-position.ts:39）・有効長 512・トークナイザ。max_distance だけ違う umT5 は容器がバイト同一になり、区別できない（実測）
- 追加学習版の fp8 に埋め込まれた spiece_model の sha256 は e3909a67… で、google/umt5-xxl と Wan pin の spiece.model（4,548,313 B）と完全一致。この利用例では「語彙数が同じで別のトークナイザ」の沈黙の誤りは起きない
- 今の書き手は F32 の safetensors（index か単一）と config.json しか読まない（umt5_export.py:112-113, 304-306）。追加学習版は BF16/FP8 の ComfyUI 単一ファイルで config 無し・キーは shared.weight。google は pickle の .bin 6 本（デコーダ込み）。どちらもそのままでは読めない
- google/umt5-xxl の config.json は Wan pin の text_encoder config と、グラフに効く欄が全部一致（vocab 256384・d_model 4096・d_ff 10240・24 層・64 head・d_kv 64・buckets 32・max_distance 128・eps 1e-6）。違うのは architectures・model_type 欠落・版だけ
- コミュニティの追加学習版 umT5 は HF 上でライセンスを宣言していない（cardData 無し・tags は not-for-all-audiences だけ）。配布の門は provenance の license を要求するので、第三者の経路を作っても license 欄が埋まらない
- 差し替え席は差し替え先の provenance・license・pipeline 名を検査も記録もしない（components.ts:416-438 は manifest と graph sha だけを見る）。案 A の出所の門は配布時にしか効かず、実行時の差し替えには効かない
