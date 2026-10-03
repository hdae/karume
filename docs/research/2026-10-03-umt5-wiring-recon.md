> この文書は 2026-10-03 時点の調査記録で、正本ではない。

# umT5 の結線 — 段 10d の下見（配布リポ・越境参照・API・text 段・signal）

読み取りだけの調査（HEAD `cc030de3`・GPU 不使用・実重みは読まない・リポのコード変更なし）。ADR 0119 の裁定 1
（umT5 は別の配布リポ — 2026-10-03 利用者）を受けて、段 10d（パイプラインへの結線）の設計の材料を集める。
事実には `file:line` を付け、推測は「推測」と書く。数値は手元の実ファイルか既存の記録の値。

## 要約

1. **越境参照は hub に実装済み**。manifest の `FileRef` の `repo` + `revision`（40 桁の commit SHA だけ）を parse が門にし、
   容器は part 全部が同じ対で揃う。HF 取得元は宣言された座標をそのまま開き、ローカル取得元は `crossRepo` の明示 mapping だけで解く
   （ローカルでは revision の値を見ない）。
2. **焼く側も実装済み**（`dist.py` の `--ref-*` 5 指定）。ただし「自リポで組むはずのバイト列と同一の役割を別リポの参照へ差し替える」
   形で、参照先の配布形がローカルに組み上がり、SHA が確定していることが前提。前例は anima-extra → anima の 1 組だけ。
3. **部品だけの配布リポの前例は無い**。`karume/5` は model ごとに `pipeline`・空でない `quants`・`defaultQuant` を必須にするので、
   umT5 リポも pipeline 名と quant 席を 1 つ名乗る。越境参照の読み手は参照先の `karume.json` を読まない。
4. **推奨の形は (b)**: Wan の manifest が `text_encoder` の容器を umT5 リポへ越境参照し、`fromPretrained` は source 1 つで両方を取る。
   資産の経路は構築時の明示の選択。トークナイザ資産は Wan リポの自前の資産に残す（資産は常に全数を取るので、越境にすると
   資産の経路まで umT5 リポに触る）。
5. **(b) の費用**: Wan の全 quant の `requiredLimits` が語彙埋め込みの 1 バッファ（1,050,148,864 B）まで上がる（資産の経路も含む —
   推計）・公開順序の制約・未公開期間の SHA の扱い。
6. **text 段の部品は揃っている**（`WanPromptEncoder`・バケット表・`session-io`）。結線の残りは admission・入口のトークナイズ・
   text 段の Session の張り畳み・経路の分岐。既存の構築の呼び出し 5 箇所は、GPU 経路を既定にした瞬間に黙って経路が変わる。
7. **generate の `signal` は画像・音声の系列に前例が無い**（anima / irodori / sbv2 は構築の `signal` だけ）。生成中の前例は
   `generation/sequence.ts`（LLM）の「run の直前で検査」。参照門は「どれか 1 本に行があれば緑」のまま（決定 8 の前提は未着手）。

## 1. 越境参照の現状（観点 1）

### 1.1 文書の取り決め

- ADR 0038 追記 2026-08-25: `FileRef` の optional `repo` / `revision`。`revision` は **40 桁小文字 hex の commit SHA だけ**・2 つは
  両方同時にだけ現れる・`size` / `sha256` は必須（二重 pin）・取得層は宣言された (repo, revision) から取る・同一性キーは
  `repo@revision/path`。焼く側は `dist.py` の 5 指定。公開順序は release-runbook §0（`docs/decisions/0038-manifest-v1.md:520-541`）。
- ADR 0109 決定 3: 越境参照は**容器単位** — `container.parts` の全要素が同じ (repo, revision)（または全部が自リポ）であることを
  parse で門にする（`docs/decisions/0109-manifest-v5-container.md:92-93`）。決定 4: 資産（`assets`）は quant 非依存の席として残り、
  越境参照も実在する（同 `:97-100`）。
- ADR 0086 決定 3: 越境の解決は「`LocalDirectoryOptions.crossRepo` の明示 mapping → 無ければ fail loudly」の 2 段。隣接する同名
  ディレクトリの推測と、未 mapping の暗黙のリモート降格は MUST NOT（`docs/decisions/0086-distribution-source.md:109-129`）。
  `fallback` 欄は 0.13.0 で撤去済み（同 `:118-120`）。

### 1.2 実装

| 層             | 事実                                                                                                                              | 根拠                                                       |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| parse          | `COMMIT_SHA_RE = /^[0-9a-f]{40}$/`・片方だけの宣言は fail loudly・repo は `owner/name` の許可リスト                               | `packages/hub/src/manifest.ts:113`・`:535-566`・`:524-533` |
| parse          | 容器の part は part 0 と同じ越境の対でなければ落ちる                                                                              | `packages/hub/src/manifest.ts:716-728`                     |
| 同一性キー     | `fileRefKey` = 自リポは `path`・越境は `repo@revision/path`                                                                       | `packages/hub/src/manifest.ts:264-272`                     |
| HF 取得元      | `originFor` は宣言の (repo, revision) で同じアダプターを開く。キャッシュは内容キーで、別リポの同名 path は別エントリ              | `packages/hub/src/sources/hf.ts:234-238`                   |
| ローカル取得元 | `originFor` は `crossRepo[repo]` だけを見る。無ければ `crossRepo` に何を渡すかを言って落ちる                                      | `packages/hub/src/sources/local.ts:104-109`・`:206-212`    |
| ローカル取得元 | `pin(_generation, …)` は世代を受けて捨てる — 越境先のローカル取得元は**宣言 revision の値を見ない**（検証は size の厳密一致だけ） | `packages/hub/src/sources/local.ts:242-247`・`:185-194`    |
| テスト         | ローカル + ローカル越境・ローカル + HF 越境・未 mapping の文言                                                                    | `packages/hub/tests/local_test.ts:349`・`:383`・`:401-412` |
| 消費側の取得   | 部品は渡した `componentKeys` だけを温める。manifest の `assets` は**常に全数**を取る                                              | `packages/models/src/hub/components.ts:462-489`            |
| 選択           | `resolveSelection` の assets は quant にも weights の絞り込みにも依らず全数                                                       | `packages/hub/src/resolve.ts:28`・`:112`・`:139`           |
| examples       | `--source-map owner/name=<パス>` を `denoDirectory(…, { crossRepo })` へ写す共通の口がある                                        | `examples/shared/local-source.ts:29-68`                    |

- **ローカルの二重 pin の扱い**: SHA の欄そのものは parse が 40 桁 hex を要求する（無い・短いは fail loudly）。ローカルの越境先は
  revision を使わないので、ローカルで回す限り「どの 40 桁 hex か」は読みに効かない。ローカル専用の緩和（SHA なしの越境）は無い。
- **前例**: `models/karume-anima-extra/karume.json` の 2 モデルが `hdae/karume-anima@adb9dcf0…` の `shared/text_encoder`（7 part）・
  `shared/vae_decoder`（3 part）・`tokenizer`・`tokenizer_2` を越境参照している（手元の manifest を読んだ値）。

### 1.3 焼く側（`karume.dist`）

- `ExternalComponents`（repo・revision・参照元のローカル dist・参照元のモデル名・役割名）。用途は「同じバイト列を 2 つのリポへ
  二重に上げない」こと（`tools/exporter/src/karume/dist.py:1689-1733`）。
- 参照先の path は参照元の `karume.json` が宣言する `<model>/<rel_path>` か `shared/<rel_path>` から引き、`size` / `sha256` は
  参照元の実ファイルから採り、さらに**自分で組むはずだったバイト列と sha256 が一致すること**を確かめる（同 `:1752-1804`）。
  分割形の容器は part ごとに 1 参照（同 `:1807-1858`）。5 指定は全部か無しか（同 `:1861-1904`）。
- 帰結: Wan の計画は umT5 の容器を自分の artifact として持ち（系列 `outputs/series/wan2.1-umt5-i8-dyn/text_encoder/`）、umT5 リポ側と
  **同じ相対 path**（モデルのサブツリーの中）に置く必要がある。`--ref-revision` は 40 桁 hex の形しか検査しない（同 `:1723-1733`）。
- 多リポの前例はこの anima-extra の 1 組だけ。SBV2 の BERT（`text_encoder`）と anima の text stack は、どちらも自リポの `shared/` に
  置かれている（`models/karume-sbv2-jvnv/karume.json`・`models/karume-anima/karume.json` を読んだ値）。

## 2. 部品だけの配布リポ（観点 2）

### 2.1 manifest の要件

- model ごとに `pipeline` は必須で `<name>/<major>`（`packages/hub/src/manifest.ts:105`・`:972-978`）。`weights` と `quants` は
  空を許さない・`assets` は空を許す・`defaultQuant` は必須（同 `:1052-1093`）。部品だけのリポも pipeline 名と quant 席を名乗る。
- quant の `weights` は完全写像（`complete_quant_weights` — ラベルが 1 つの weights は機械が埋める —
  `tools/exporter/src/karume/dist.py:643-653`）。
- **越境参照の読み手は参照先の `karume.json` を読まない**（Wan の manifest の FileRef が座標を全部持つ）。部品差し替え席
  （`components`）も、参照先の manifest の pipeline 名は見ずにグラフ記述の sha256 だけを突き合わせる
  （`packages/models/src/hub/components.ts:416-438`）。umT5 リポの pipeline 名を解釈する TS 実装は、(b) の形では要らない。
- 逆に umT5 リポを `WanPipeline.fromPretrained` に直接渡すと、Wan の admission が pipeline 名の不一致で落ちる
  （`packages/models/src/wan/pipeline.ts:871-884`）— 取り違えは既存の門で拾える。

### 2.2 中身と規模（手元の実ファイル）

| 物                 | 値                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| umT5 の容器        | 26 part・計 5,686,978,107 B（5.296 GiB）・最大の part 268,435,456 B・グラフ名 `text_encoder`           |
| 語彙埋め込み（i8） | `[256384, 4096]` = 1,050,148,864 B + scale 1,025,536 B（調査 2026-10-03-umt5-encoder-recon §1）        |
| トークナイザ資産   | `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json` 8,117,546 B                                      |
| Wan の配布形（今） | transformer f16 2,841,678,093 B・i8 1,426,806,948 B・VAE 2 本 289,940,137 B・`text_embeds` 5,280,288 B |

- 容器の出所は `Wan-AI/Wan2.1-T2V-1.3B-Diffusers@0fad780a534b6463e45facd96134c9f345acfa5b` の `text_encoder`（Apache-2.0 —
  `tools/export-recipes/wan/sources.py:55-59`・`tools/export-recipes/wan/umt5_export.py:500-505`）。グラフ名 `text_encoder` は
  仮置きで、部品名と書く関数・`tests/test_graph_names.py` の `ENTRIES`・`WAN_WEIGHTS` の行は 10d で足す
  （`tools/export-recipes/wan/umt5_export.py:36-45`）。
- トークナイザ資産の形式名は `karume-wan-umt5-tokenizer/1` で、umT5 の T5 の表と Wan（diffusers）の前処理 `prompt_clean` の表を
  1 本に束ねている（`packages/models/src/wan/text/tokenizer.ts:40`・`:49-52`・`:118-131`）。出所は Wan-AI リポの `tokenizer/`
  （`tools/export-recipes/wan/umt5_tokenizer.py:145-161`・`:297`）。
- **google/umt5-xxl との関係**: 確認済みなのは `spiece.model` の一致（sha256 `e3909a67…`）と config の `_name_or_path` だけ
  （調査 2026-10-03-umt5-encoder-recon §1.1・§1.4）。**重みの一致は未確認**。Wan が umT5 を学習中に凍結していたかも、この
  リポの中では確かめていない。

### 2.3 リポ名（ADR 0092 決定 2）

- 規則は `karume-<family>[-<世代 or 版>][-<変種>]`（`docs/decisions/0092-distribution-repos-and-sources.md:58-68`）。umT5 を家族、
  XXL を変種と読むと `karume-umt5-xxl`。
- 名前が汎用（`karume-umt5-xxl`）で良いかは、重みが google/umt5-xxl の encoder と同じかで決まる。違うなら「Wan の umT5」であって、
  汎用の名前は誤った約束になる（推測 — 2.2 の未確認の事実に掛かる）。

### 2.4 カード・NOTICE・LICENSE

- Wan の NOTICE は「The text encoder is not distributed」と書いている（`tools/export-recipes/wan/distribution.py:592-594`）。
  (b) では Wan の配布形が umT5 を参照するので、この文面は書き換えが要る。
- 配布リポ直下の `LICENSE.md` / `NOTICE.md` は `Pipeline.root_files` で書く（ADR 0092 決定 7・`distribution.py:613-616`）。umT5 リポは
  上流が同じ Apache-2.0 なので、Wan の `apache_license_2_0()` をそのまま使える。改変の告知（i8 per-channel・相対位置の表は F32・
  `gelu_new` → `gelu_tanh`・有効長 L の動的形・バケット表はグラフ入力）は umT5 リポの NOTICE の新規の文面になる。
- カードの描き手は `Pipeline.card_profiles`（`tools/exporter/src/karume/dist.py:1907-1928`）。部品だけのリポのカードの Usage に
  載せる TS のクラスが無い（umT5 単体の公開クラスは無い）。

## 3. WanPipeline の API 案（観点 3）

### 3.1 今の構築の流れ

`fromPretrained` は `loadManifest` → `resolveSelection` → `loadContainerComponents`（descriptor だけ取る → 家族 admission →
重みの part → 資産の全数）→ `#build`（`packages/models/src/wan/pipeline.ts:784-827`）。部品のキーは `transformer` /
`vae_decoder_first` / `vae_decoder_next` の 3 本（同 `:141-144`）で、`text_embeds` は manifest の資産（同 `:147`・`:948-951`）。
部品差し替え席 `components` が既にある（`packages/models/src/hub/load-options.ts:56-70`）。

### 3.2 3 案

| 案  | 形                                                                                                                                                   | 守るもの                                                                                                                           | 費用                                                                                                                                                                                                                     |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| (a) | Wan の manifest は umT5 を知らない。`fromPretrained(source, { textEncoder: { source: <umT5 のリポ / ローカルの取得元> } })` で渡し、省けば資産の経路 | Wan の quant の `requiredLimits` が上がらない・公開順序の制約が無い                                                                | 省いたときの既定が資産の経路になり、決定 7（GPU 経路が既定）と食い違う。取得元の既定席は作らない（ADR 0092 決定 3）ので、既定を GPU にすると source を 2 つ綴らせることになる。Wan と umT5 の組み合わせを誰も pin しない |
| (b) | Wan の manifest が `text_encoder` の容器を umT5 リポへ越境参照する。`fromPretrained` は source 1 つで両方を取る。資産の経路は構築時の明示の選択      | 既定 = GPU 経路（source 1 つで足りる）・組み合わせの再現性（Wan の revision が umT5 のバイト列を二重 pin）・既存の機構だけで足りる | 公開順序（umT5 → SHA → Wan）。Wan の全 quant の `requiredLimits` が上がる。未公開期間の SHA の扱い（5 章の軸 E）                                                                                                         |
| (c) | 別クラス `WanTextEncoder` を利用者が組んで、埋め込みを `generate` に渡す                                                                             | 部品の独立性が最大                                                                                                                 | 公開面が増える（ADR 0008 の薄い面）。既定 = GPU 経路にならない。Session の順序（text を DiT の前に畳む — 決定 11）を利用者に預ける                                                                                       |

- **推奨 (b)**。決め手は 3 つ。① 決定 7 の「GPU 経路が既定」を、source 1 つの呼び出しのまま満たせるのは (b) だけ。② 同じ Wan の
  revision なら umT5 のバイト列も固定される（越境参照の二重 pin — 「同じ文字列で経路も値も黙って変わらない」を配布形が保証する）。
  ③ hub 側の追加が要らない — HF とローカルの越境解決・容器単位の門・部品差し替え席（umT5 を別の出所へ差し替える口）が揃っている。
- **(b) の資産の経路の取得量**: 部品は `componentKeys` で選ぶので、資産の経路は umT5 の容器を 1 バイトも取らない
  （`components.ts:462-484`）。ただし資産は全数を取るので（`components.ts:486-489`）、GPU 経路は `text_embeds` 5.3 MB を、
  資産の経路はトークナイザ 8.1 MB を余分に取る（今の配布形 2.92 GiB の 0.3% 以下）。
- **(b) で資産を越境にしてはいけない理由**: トークナイザを umT5 リポへの越境参照にすると、資産は全数を取るので資産の経路まで umT5
  リポに触る。ローカルでは `crossRepo` を渡さないと落ち（`local.ts:206-212`）、「umT5 無しで使える」がローカルで崩れる。

### 3.3 (b) の `requiredLimits`（推計 — 規則から導いた値で、焼いてはいない）

- 焼き方は「quant が選ぶ全部品の、常駐 1 バッファの最大バイト数」（`tools/exporter/src/karume/dist.py:722-760`）。piece 列に
  割れた重みも GPU 側では親 1 本のバッファに戻るので親の大きさで数える（同 `:686-710`）。既定の上限は
  `maxBufferSize` 268,435,456・`maxStorageBufferBindingSize` 134,217,728（`tools/exporter/src/karume/limits.py:59-62`）。
- 今の Wan の 3 席は `requiredLimits` を持たない（`models/karume-wan2.1/karume.json` を読んだ値）。quant の weights は完全写像なので、
  `text_encoder` を足すと**全席**が語彙埋め込み 1,050,148,864 B を需要に持ち、2 つの上限が両方とも焼かれる見込み。資産の経路
  （umT5 を開かない）も、取得の前の門（`pipeline.ts:811-817`）でこの値を要求される。
- 解消の道は段 10e の「語彙埋め込みのホスト gather」（ADR 0119 決定 5）。i8 の最大の linear は `[10240, 4096]` で 41,943,040 B なので、
  gather にすれば需要は既定の内に戻る見込み（推計）。

## 4. text 段の結線（観点 4）

### 4.1 部品（揃っているもの）

| 部品               | API                                                                                                                                      | 根拠                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| トークナイザ資産   | `parseWanTokenizerAsset(raw)` → `new WanPromptEncoder(assets)` → `encode(prompt, label)` が `Int32Array [L]`（L は 2〜512・末尾 `</s>`） | `packages/models/src/wan/text/tokenizer.ts:58-112`・`:118-131`                                      |
| 資産の JSON の読み | `readAssetJson(where, what, assets, key)`                                                                                                | `packages/models/src/hub/asset-readers.ts:93`                                                       |
| バケット表         | `buildUmt5RelativePositionBuckets(L)` → i32 `[L, L]`（最大 512）                                                                         | `packages/models/src/wan/umt5/relative-position.ts:45`・`:72`                                       |
| 入力の束           | `umt5SessionInputs(ids, buckets)` → `input_ids [1,L]` / `relative_position_buckets [L,L]`（2 本の L の一致を見る）                       | `packages/models/src/wan/umt5/session-io.ts:21-61`                                                  |
| 出力の詰め         | `padUmt5Context(output, tokens, rows, width)` → `[rows·width]` f32（行数・幅・dtype を見る）                                             | `packages/models/src/wan/umt5/session-io.ts:70-96`                                                  |
| 段の Session       | `withSession`（張る → 回す → 畳む）・anima の `withStage`（`stage` イベントで挟む）                                                      | `packages/models/src/session/with-session.ts:29`・`packages/models/src/anima/pipeline.ts:1202-1214` |

### 4.2 流れ（案）

1. **入口（GPU の前）**: `planWanGeneration` の GPU 経路の枝で、positive と negative を `WanPromptEncoder.encode` に掛ける。
   拒否は全部ここで `ModelInputError`（`PromptCleanError` を含む）。anima も「プロンプト層」を GPU の段より前に置いている
   （`packages/models/src/anima/pipeline.ts:1216-1221`）。negative を省いたときの既定の文字列は、資産の `negative` の行の原文
   （公式の `sample_neg_prompt`）を GPU で回す形が自然（推測 — 既定の negative の出所は 10d で決める）。
2. **text 段**: umT5 の Session を張り、positive → negative の 2 回を回し、畳む。`stage` イベントの `component` に `text_encoder` を
   足す（今の union は `transformer` / `vae_decoder` — `pipeline.ts:232-236`。`WanRunComponent` も同じ — `:224`）。Session の実行
   オプションは `{}`（VAE と同じく quant の `session` を受けない — `pipeline.ts:1153-1155`。i8 の重みは格納から決まる）。
3. **DiT 段**: `#denoise` の `positive` / `negative` を、資産の経路では `padWanTextEmbedding`、GPU 経路では `padUmt5Context` の結果に
   する（分岐点は `pipeline.ts:1048-1051` の 2 行だけ）。
4. **畳む順**（決定 11）: text 段の Session を畳んでから DiT の Session を張る。今の DiT → VAE と同じ `disposeSteps` の形
   （`pipeline.ts:1111-1123`）。切り替えの残りは fdinfo で記録（ADR 0119 未解決）。

### 4.3 admission で見る契約（重みを取る前・宣言だけで判る）

- `text_encoder` のグラフ入力が `input_ids`（i32・`[1, L]`）と `relative_position_buckets`（i32・`[L, L]`）で、**同じ記号 L**。
- 出力が 1 本で `[1, L, W]`、W が DiT の `encoder_hidden_states` の幅（`ditContract` の `contextWidth` — `pipeline.ts:665-669`）と一致。
- 最大長 512（`WAN_UMT5_MAX_LENGTH`）が DiT の文脈の行数（`contextRows`）以下。トークナイザの `maxLength` も同じ値であること
  （トークナイザは資産なので `#build` で見る — 資産は admission の時点で届いていない、`pipeline.ts:854-856`）。
- DiT の契約の検査（`ditContract` — `pipeline.ts:636-670`）と同じ書き方で、`export` して GPU 無しのテストで縛れる。

### 4.4 既存の資産の経路との分岐点と、黙って経路が変わる呼び出し

- 分岐は 3 か所: `#admit`（`text_encoder` を開くか）・`#build`（`text_embeds` を解析するか / トークナイザを解析するか —
  `pipeline.ts:948-951`）・`planWanGeneration`（引くか / トークナイズするか — `pipeline.ts:402-415`・`:447-465`）。
- `WanPipeline.prompts`（`pipeline.ts:997-1004`）は資産の経路の受理集合。GPU 経路での意味は 10d で決める（未決）。
- **構築の呼び出しは 5 箇所**: `examples/wan/main.ts:103`・`tools/gpu-lab/browser/wan-tab.ts:400`・
  `packages/models/tests/e2e_wan_pipeline_test.ts:289`・`:442`・`packages/models/tests/e2e_wan_ab_test.ts:194`。GPU 経路を既定に
  した時点で、明示しない呼び出しは全部 GPU 経路へ移る。資産の経路の既存の sha 行 8 本（`fixtures/references/wan.json` — 全部
  B570 の環境キー）を守るには、sha 行と A/B の呼び出しに資産の経路の明示が要る。
- examples/wan はプロンプトを資産の行の名前で受ける（`examples/wan/main.ts:24`・`:106-110`）。`--source-map` も渡していない
  （`main.ts:84-94` は `distributionSource(source)` を mapping なしで呼ぶ）。
- gpu-lab のブラウザのタブは HTTP の取得元で読むので、越境先（umT5 リポの座標）もこのサーバが HF の形で配る必要がある（推測 —
  サーバの実装は読んでいない）。段 9（Chrome）と重なる。

### 4.5 sha 行と参照門

- 参照門 `referenceGatePasses` は今も「登録した case id のどれか 1 本に行があれば緑」（`packages/runtime/tests/helpers/reference.ts:301-306`）。
  決定 8 の前提（新しい case id を足す前に締める）は未着手。
- 利用者は RTX 5070 Ti で試す予定。今の Wan の sha 行は B570 の環境キー（`deno-intel-graphics-bmg-g21`）だけなので、5070 Ti では
  sha のケースは明示 SKIP + 参照門が赤になり、`KARUME_REFERENCE=write` で自機の行を作る運用（ADR 0106）。

### 4.6 拒否の文言（裁定 3 — 直し方まで言うか）

| 拒否                       | 直し方を言うか                                                   | 根拠                                                           |
| -------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------- |
| entity の候補              | 言う（「`&` の直後に空白を入れるか、参照を元の文字に直す」）     | `packages/models/src/wan/text/prompt-clean.ts:236-242`         |
| mojibake                   | 言う（「元の文字に直してから渡す」）                             | 同 `:254-262`                                                  |
| 未割り当てのコードポイント | **言わない**（理由だけ）                                         | 同 `:293-299`                                                  |
| C1 制御文字                | **言わない**（理由だけ）                                         | 同 `:300-306`                                                  |
| 特殊トークン・空白直後の ▁ | 言う                                                             | `packages/models/src/wan/text/tokenizer.ts:88-102`             |
| 語彙外・上限超え           | 言う（「その文字を除くか言い換える」・「プロンプトを短くする」） | `packages/models/src/text/t5-tokenizer.ts:139-142`・`:161-165` |
| 下限割れ（L = 1）          | 言う（「1 語以上入れる」）                                       | `packages/models/src/wan/text/tokenizer.ts:103-108`            |

- entity の文言は `R&D` → `R & D` のような具体例を持たない。裁定 3 の「直し方まで」を満たすには、未割り当てと C1 に「その文字を除く」を
  足し、entity に例を足すのが最小（10d の小物）。

## 5. signal（観点 5 — 決定 9）

| 系列 / 層                     | 構築の `signal`                                           | 生成中の `signal`                               | 根拠                                                                                                      |
| ----------------------------- | --------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| anima                         | `AnimaPipelineOptions.signal`（取得層と構築の両方へ渡す） | 無い                                            | `packages/models/src/anima/pipeline.ts:355-366`・`:892-899`・`:955`・`:1039-1070`                         |
| irodori                       | `IrodoriPipelineOptions.signal`（同上）                   | 無い                                            | `packages/models/src/irodori/pipeline.ts:353-356`・`:865-872`・`:918`・`irodori/admission.ts:319`・`:365` |
| sbv2                          | 構築オプションの `signal`                                 | （読んでいない）                                | `packages/models/src/sbv2/pipeline.ts:202`                                                                |
| generation/sequence.ts（LLM） | —                                                         | 要求の `signal`・各 run の発行直前に検査        | `packages/models/src/generation/sequence.ts:403-404`・`:1089`・`:1113`・`:1130`・`:1368-1372`             |
| gemma chat                    | —                                                         | 要求の `signal`・`signal.reason` を包まず投げる | `packages/models/src/gemma/pipeline.ts:546`・`:1398-1402`                                                 |

- 共通の道具は `settleAbort`（`setTimeout(0)` で 1 度譲ってから `throwIfAborted` — マイクロタスクでは中断のタスクを観測できない）
  （`packages/models/src/concurrency/abort.ts:8-31`）。
- `hubLoadOptions` は `signal` を写すが、置き場（`XPipelineOptions` か `XFromPretrainedOptions` か）は家族が決める
  （`packages/models/src/hub/load-options.ts:10-12`・`:81-90`）。`loadContainerComponents` は取得の後の Session の読みへ `signal` を
  持ち越さない（`packages/models/src/hub/components.ts:452-459`）。
- ADR 0083 決定 5 は「cancel は `AbortSignal` が正（`onEvent` の throw を中断手段にしない）」（`docs/decisions/0083-generation-api-surface.md:122`）。
- **事実の補正**: ADR 0119 決定 9 と limitations の Wan 節（`docs/limitations.md:1004-1006`）は「他の系列は `signal` を持つ」と書くが、
  画像・音声の系列が持つのは**構築**の `signal` で、生成中の `signal` を持つのは LLM の層だけ。Wan の `generate` の `signal` は
  `sequence.ts` の形（境目で検査・包まず投げる・後始末は本体の失敗を上書きしない）を写すことになる。
- Wan に入れる位置（案）: 構築は `WanPipelineOptions.signal`（anima / irodori と同じ — 取得層と `#admit` / `#build` の境目）。
  生成は `WanGenerateRequest.signal` で、`planWanGeneration` の後・text 段の前後・DiT の各 step の間・VAE の段の前（タイルの間に
  置くかは 10d で決める）。中断時は開いている Session を `disposeSteps` で畳んでから `signal.reason` をそのまま投げる。

## 6. 影響ファイルと段の分割（観点 6）

| 段    | 中身                                                                                                                                                                                          | GPU                                  |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| 10d-1 | umT5 の配布リポの recipe（`Pipeline`・計画・NOTICE / LICENSE・カード）・`dist.py` の 1 行・Wan の計画に `text_encoder` の i8 とトークナイザ資産・越境参照で組む・`verify_dist`・両方の pytest | 不要                                 |
| 10d-2 | `WanPipeline` の結線（経路の選択・admission の契約・入口のトークナイズ・text 段・`stage` / 診断の名前）・既存の呼び出し 5 箇所の明示・examples の `--source-map` と自由プロンプト             | 単体は不要・e2e と VRAM の記録は要る |
| 10d-3 | `signal`（構築と生成）・中断の後に次の generate が回ること                                                                                                                                    | 構築側は不要・生成中の検査は要る     |
| 10d-4 | 参照門を締める（先に・runtime の横断変更）→ GPU 経路の新しい case id の sha 行・品質の記録（3 点比較）・視認素材の PNG（`contactSheet` — `e2e_wan_pipeline_test.ts:972-975`）・文書           | 要る                                 |

- 順序の制約: 10d-1 の後に 10d-2（配布形が無いと e2e が SKIP）。10d-4 の参照門は sha 行を足す前。10c の GPU の移植の門（帯の導出）は
  まだ実走していない（ADR 0119 追記の段 10c の結果 — 「GPU の実走は視認素材の生成の後」）ので、10d-2 の e2e はその後。
- 段 9（Chrome）との重なり: `pipeline.ts` と `tools/gpu-lab/browser/wan-tab.ts`。

## 設計の軸

### 軸 A — umT5 への参照の形

- 案: (a) 構築オプションで umT5 の source を受ける / (b) Wan の manifest の越境参照 / (c) 別クラス。
- **推奨 (b)**（3.2）。

### 軸 B — 資産の経路の選択ノブ

- 案: B1 構築時のオプション（例 `textEncoder: "gpu" | "precomputed"`・既定 `"gpu"`）/ B2 生成の要求ごと / B3 quant 席で表す。
- **推奨 B1**。経路で取得する部品が変わる（資産の経路は umT5 を取らない）ので、取得の前に決まる構築時が自然。B2 は両方を取る前提に
  なり、軽い使い方が消える。B3 は quant の weights が完全写像なので「umT5 を持たない席」を表せない（`dist.py:643-653`）。名前は未決。

### 軸 C — トークナイザ資産の置き場

- 案: C1 umT5 リポに置いて越境参照 / C2 Wan リポの自前の資産 / C3 T5 の表（umT5 リポ）と `prompt_clean` の表（Wan リポ）に割る。
- **推奨 C2**。資産は常に全数を取るので、C1 は資産の経路まで umT5 リポに触る（3.2）。形式は Wan の前処理を束ねた
  `karume-wan-umt5-tokenizer/1` で、出所も Wan-AI リポ（2.2）。C3 は 10a の資産の形式を割り直す費用がかかり、他の消費者がまだいない。
  umT5 リポを他の系列が使う日に、汎用のトークナイザ資産を umT5 リポへ足す。

### 軸 D — umT5 リポの名前・モデル名・pipeline 名

- 名前: `karume-umt5-xxl`（汎用）か、Wan の写しであることを名乗る名前か。**決め手は重みが google/umt5-xxl の encoder と同じか**（未確認 —
  2.2）。推奨: 先に確かめ、同じなら `karume-umt5-xxl`。
- pipeline 名: 読む TS 実装が無い（2.1）ので、何を名乗っても今は解釈されない。推奨は部品の役を名乗る `umt5-encoder/1`（`PIPELINE_RE`
  に収まる — `manifest.ts:105`）。quant 席は `i8` 1 つ。

### 軸 E — 未公開期間の開発（SHA が無い間）

- 案: E1 仮の 40 桁 hex（例 全ゼロ）で Wan を越境参照で組み、ローカルは `crossRepo` で解く（ローカルは revision を見ない — 1.2）/
  E2 開発中の Wan は umT5 を自リポに持つ自己完結の形で組み、公開時に越境へ切り替える / E3 umT5 リポを先に公開して本物の SHA を得る。
- **推奨 E1 + 機械の門**。e2e が公開と同じ manifest の形（越境参照）を読む。仮の SHA が公開へ漏れる危険は、今は release-runbook の
  順序（§0）だけが防ぐので、仮の SHA を拒む門（組み立てか公開の手順）を足す。E2 は e2e が公開と別の形を読み、5.3 GiB の写しも持つ。
  E3 は公開（HF への上げ）を前倒しする判断で、利用者が決めること。

### 軸 F — text 段の Session の寿命

- 1 案: generate ごとに張って畳む（決定 11）。常駐させる案は B570 で DiT 段（81 フレームで 7.31 GiB）と同居できない（ADR 0119
  決定 11）。費用は generate ごとの 5.3 GiB の読み直しで、所要は未測。

### 軸 G — `signal` の置き場

- 1 案: 構築は `WanPipelineOptions.signal`（anima / irodori と同じ）、生成は `WanGenerateRequest.signal`（`sequence.ts` の形）（5 章）。

## 影響ファイル

| 区分            | ファイル                                                                                                                                                                                                                                          | 段           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| recipe（配布）  | 新規の umT5 の配布 recipe（置き場は未決 — `tools/export-recipes/wan/` の中か新しい家族の席）・`tools/export-recipes/dist.py`（`PIPELINES` に 1 行）                                                                                               | 10d-1        |
| recipe（Wan）   | `tools/export-recipes/wan/distribution.py`（`WAN_OUTPUT_PATHS`・`WAN_WEIGHTS`・`WAN_ASSETS`・計画の検査・NOTICE）・`wan/card.py`・`wan/umt5_export.py`（書く関数）                                                                                | 10d-1        |
| recipe のテスト | `tools/export-recipes/tests/test_graph_names.py`（`ENTRIES`）・`tools/export-recipes/wan/tests/test_distribution.py`                                                                                                                              | 10d-1        |
| パイプライン    | `packages/models/src/wan/pipeline.ts`・`text-embeds.ts`・`config.ts`（必要なら）・`packages/models/wan.ts`（公開面の doc）                                                                                                                        | 10d-2・10d-3 |
| 文言            | `packages/models/src/wan/text/prompt-clean.ts`（未割り当て・C1・entity の例）                                                                                                                                                                     | 10d-2        |
| 呼び出し        | `examples/wan/main.ts`・`examples/wan/README.md`・`tools/gpu-lab/browser/wan-tab.ts`（とそのサーバ）                                                                                                                                              | 10d-2        |
| テスト          | `packages/models/tests/wan_pipeline_test.ts`・`e2e_wan_pipeline_test.ts`・`e2e_wan_ab_test.ts`・`fixtures/references/wan.json`                                                                                                                    | 10d-2〜10d-4 |
| 参照門          | `packages/runtime/tests/helpers/reference.ts`                                                                                                                                                                                                     | 10d-4 の前   |
| 文書            | `docs/limitations.md`（Wan 節）・`docs/release-runbook.md`（§0 の越境の組と Wan の初公開の節 — `:237-259` は「越境参照なし」と書く）・`docs/glossary.md`・`docs/assets-layout.md`・`docs/backlog.md`・ADR 0119 の追記・`.claude/ACTIVE_DESIGN.md` | 各段         |

## 未解決（オーケストレータ / 利用者の判断）

1. Wan の `text_encoder` の重みは google/umt5-xxl の encoder と同じか（リポ名を決める事実 — 軸 D）。
2. 未公開期間の SHA（軸 E）と、仮の SHA を公開へ漏らさない門の置き場。
3. umT5 リポを `*_SOURCES` / `KARUME_SOURCES` と疎通テスト（`tools/published-smoke`）のどこに載せるか — TS の家族を持たないリポで、
   ADR 0092 決定 3 の「キー = リポ名から `karume-` を落としたもの」の表の行き先が無い。
4. 資産の経路にも `requiredLimits`（約 1.05 GB の binding）が掛かることを、段 10e のホスト gather まで受けるか。
5. GPU 経路での `prompts` の意味と、negative を省いたときの既定の文字列の出所（資産の原文か、定数か）。
6. umT5 リポのカードの Usage に何を載せるか（umT5 単体の公開クラスは無い）。
7. 生成中の `signal` を VAE のタイルの間にも置くか。

## 出典

- ADR: `docs/decisions/0119-wan-umt5-gpu-text-encoder.md`（決定 5・7・8・9・10・11・裁定 2026-10-03・追記 10a / 10b / 10c）・
  `0038-manifest-v1.md:520-559`・`0109-manifest-v5-container.md:92-100`・`:169-175`・`0086-distribution-source.md:109-129`・
  `0092-distribution-repos-and-sources.md:49-95`・`0083-generation-api-surface.md:122`
- hub: `packages/hub/src/manifest.ts`・`resolve.ts`・`sources/local.ts`・`sources/hf.ts`・`tests/local_test.ts`・`tests/container_source_test.ts`
- models: `packages/models/src/wan/{pipeline.ts,text-embeds.ts,config.ts}`・`src/wan/text/{tokenizer.ts,prompt-clean.ts}`・
  `src/wan/umt5/{session-io.ts,relative-position.ts}`・`src/hub/{components.ts,load-options.ts,asset-readers.ts}`・
  `src/concurrency/abort.ts`・`src/anima/pipeline.ts`・`src/irodori/pipeline.ts`・`src/generation/sequence.ts`・`src/text/t5-tokenizer.ts`
- exporter / recipe: `tools/exporter/src/karume/{dist.py,limits.py}`・`tools/export-recipes/{dist.py,wan/distribution.py,wan/sources.py,wan/umt5_export.py,wan/umt5_tokenizer.py}`
- 手元の実ファイル: `models/karume-wan2.1/karume.json`・`models/karume-anima-extra/karume.json`・`models/karume-anima/karume.json`・
  `models/karume-sbv2-jvnv/karume.json`・`outputs/series/wan2.1-umt5-i8-dyn/text_encoder/`（26 part）・`outputs/series/wan2.1-umt5-tokenizer/tokenizer.json`
- 文書: `docs/release-runbook.md:10-66`・`:237-259`・`docs/limitations.md:979-1015`・
  `docs/research/2026-10-03-umt5-encoder-recon.md`（§1.1・§1.4）
