# 0109: manifest `karume/5` — コンテナ配布形の入口

- Status: accepted（2026-09-22 — ユーザー裁定「それぞれ推奨案で承認」: コンテナ形式の波 段 2 の
  計画で示した推奨案 1-1〜1-12 と、再アップロードする 1 系列 = irodori-v4.1-small）
- Date: 2026-09-22
- 関連: ADR [0108](0108-container-format.md)（コンテナ形式 — 決定 20 の責務分担を本 ADR が具体化し、
  一部を訂正する〈0108 追記 2〉）/ [0038](0038-manifest-v1.md)（format 繰り上げ規則・FileRef 3 点
  セット・越境の対）/ [0041](0041-manifest-v2.md)（`quants` の語彙）/
  [0071](0071-manifest-v3-shards.md)（`shards` 欄 — 本 ADR で退役）/
  [0075](0075-quant-presentation.md)（`label` / `description` — 据え置き）/
  [0080](0080-hub-fetch-cache-050.md)（取得層の cold / warm 規律 — 継承）/
  [0086](0086-distribution-source.md)（取得元の抽象 — 継承）/
  [0089](0089-memory-limits-preflight.md)（1 バイトも取る前の見積り — 継承）/
  [0090](0090-shard-spec-v3-tensor-pieces.md)（tensor pieces — 退役）/
  [0085](0085-ple-host-gather.md)（PLE sidecar — 資産の席がコンテナへ移る）。
  仕様の正本は [container-v1](../container-v1.md)（物理形式）と本 ADR（manifest の形）。

## Context

`karume/4` は「部品 × dtype → shard 列（先頭 = グラフ shard）+ extras」「quant 席 = 部品 → dtype
ラベルの完全写像 + 実行ノブ」「model 単位の assets」の 3 層である
（`packages/hub/src/manifest.ts:259-283, :306-352`）。配布形が safetensors 方言から `krm` へ
移る（ADR 0108）ので、shard 列の席は**コンテナの入口**（descriptor と part の FileRef）へ置き
換わる。ADR 0108 決定 20 の表は入口を `quants[].container` と書いていたが、実ミラー 11 本の
棚卸し（2026-09-22・`models/*/karume.json` の集計）で次が分かった:

- 部品 × dtype の一意ファイルは **65.9 GiB**。quant 席は「重みの写像 + 実行ノブ」であって重みの
  単位ではない — gemma4 の 3 席（`i4` / `i4-gemvpar` / `i4-fast`）は**重みが同一**で実行ノブだけ
  違い、anima は 5 モデル × 6 席が同じ text_encoder（1.1 GB）を指す。
- 席ごとに 1 コンテナを物理ファイルにすると **197 GiB（×2.99）** になる（anima ×3.87・
  sbv2 ×4.66・gemma4 ×3.00・irodori ×1.22）。
- extras は anima transformer の `rope_base`（65,968 B・f16 / i8 が同じ path）**1 種だけ**。
  越境参照は anima-extra の text_encoder / vae_decoder の shard と tokenizer 2 本。PLE は gemma4 の
  `assets` に shard 9 本（計 2.3 GB）+ 索引 1 本。
- 段 1 の読み手は取得の口をもう持っている: `BlockSource`（part 数・part 長・
  `read(part, offset, length)`）を `openContainer` が受け、**2 文書それぞれ**の期待 length + sha256
  で照合してから parse する（`packages/runtime/src/format/container/open.ts:44-61`）。
- 取得層（`@hdae/fetch-cache` 0.8.0）に HTTP Range の口は無い。cold は「ファイル全量を流しながら
  sha256 を検証し、記録ハッシュを焼く」、warm は記録ハッシュの文字列比較だけ
  （`packages/hub/src/sources/hf.ts:110-131`・ADR 0080）。区間読みは温めた後のキャッシュに
  対してだけ開き、費用は Deno の既定で scan（offset 比例）・ブラウザで seek
  （`packages/hub/src/source.ts:108-113`）。

## Decision

### 1. format は `karume/5`。`karume/4` は読まない

ADR 0038 決定 1 の規則どおり major を繰り上げ、旧 major は unsupported format で落とす。
両読みは実装しない（ADR 0108 決定 18・却下案 3）。

### 2. コンテナの粒度は**部品 × dtype**。quant 席は写像のまま

`weights.<部品>.<dtype>` が `{ container }` を持ち、`quants[].weights`（部品 → dtype ラベルの
完全写像）と `session` / `gpuFeatures` / `requiredLimits` / `label` / `description` は `karume/4` の
まま。ADR 0108 決定 20 の `quants[].container` は**この形に訂正する**（0108 追記 2）。

- 理由は上の実測: 席ごとの物理ファイルは ×2.99。段 1 の書き手（1 コンテナ 1 グラフ）と移行 CLI
  もこの粒度で書いており、そのまま噛み合う。
- 1 コンテナに複数グラフを載せる口（container-v1 §2.1 の `graphs`）は**形式の能力として残す**が、
  `karume/5` は使わない。
- dtype ラベルは選択・表示の語彙で、格納の正本は descriptor の `encoding.codec`（ADR 0071 決定 3 の
  「ラベルは格納を主張しない」を継承）。hub は写像の完全性（席の全部品にコンテナがある）だけを
  見る。

### 3. `container` 欄 — descriptor の期待値 + part の FileRef 列

```json
"container": {
  "descriptor": {
    "graph": { "length": 183422, "sha256": "…" },
    "model": { "length": 40961, "sha256": "…" }
  },
  "parts": [
    { "path": "v4.1-small/dit/model.i8-00001-of-00003.krm", "size": 224407, "sha256": "…" },
    { "path": "v4.1-small/dit/model.i8-00002-of-00003.krm", "size": 0, "sha256": "e3b0c442…" },
    { "path": "v4.1-small/dit/model.i8-00003-of-00003.krm", "size": 268435456, "sha256": "…" }
  ]
}
```

- **`descriptor` は 2 文書それぞれの length + sha256**（part 0 ファイルの sha256 とは別の事実）。
  読み手の契約（`DescriptorExpectation`）を変えずに済み、`graph` 文書の sha256 はそのまま `krg` の
  同一性（ADR 0108 決定 4 — 内容ハッシュ）に使える。
- **`parts` は part 0 を含む全 part** を添字順に並べる（container-v1 §8）。先頭が part 0 =
  descriptor のファイルで、`size ≥ 24`（ヘッダ長）。2 要素以上（part 0 + part 1）。
- **長さ 0 の part（const が空の part 1）は 0 バイトのファイルとして書き、`container.parts` に限り
  `size: 0` を許す**（sha256 は空列の値）。hub は取得しない（`BlockSource.partLength` が 0 を
  答え、読みは起きない）。FileRef 一般の `size > 0`（`manifest.ts:547`）はそのまま。
- 上限は**独立に持つ**: part 件数 ≤ 1024・1 part ≤ 1024 MiB（part 長の天井 — container-v1 §10 の
  値を hub 側の定数として持ち、Python の突合先は `tools/exporter/src/karume/container.py` の定数）。
  `karume/4` の `MAX_SHARD_BYTES` = 256 MiB とその `assets` / `extras` 免除は退役。
- **越境参照は容器単位**: `parts` の全要素が同じ `repo` / `revision` の対（または全部が自リポ）で
  あることを parse で門にする。片方だけ・混在は fail loudly。
- ファイル名の規約（`<stem>-NNNNN-of-NNNNN.krm` — container-v1 §8）は書き手の規約で、hub は
  検査しない（`karume/4` の shard と同じ）。

### 4. `assets` は残す。PLE と extras はコンテナの中へ

- **model 単位の `assets`（quant 非依存・FileRef）は残す** — tokenizer / symbols / style_vectors /
  speaker_embeddings は複数コンテナで共有され、越境参照（anima-extra）も実在する。コンテナに畳むと
  リポ間・席間で複製される。
- **PLE sidecar は `assets` から消え、モデルコンテナの `assets` へ移る**: `values` と `scales` を別々の
  行列として block 上限 32 MiB 以下・**行の倍数**で切り直し（旧 shard の境界は消える）、役割
  `ple-values` / `ple-scales` の block 列（資産名 `ple.values.<k>` / `ple.scales.<k>`）と、役割 `ple-index`
  の索引（schema 3 — 行バイト数と block ごとの token 区間）にする。区間読みを要する block は
  **1 block = 1 part**（container-v1 §4.2 の「専用 part に単独」）。索引や `rope_base` のような全量読みの
  資産は資産どうしで 1 part を共有してよい（重み block とは同居しない）。読み手は token → (block, 行 offset)
  の翻訳だけをする。
- 資産は shape を持たないので **`assets[].length`（payload のバイト数）を宣言する**（block 長 = 4 の倍数への
  切り上げ）。消費側が末尾の 0x00 を推測で剥がない。
- **`extras` は退役**し、実物 1 種（`rope_base`）はコンテナの `assets`（役割 `rope-base`）へ移る。
  複製は 66 KB × 2 本（f16 / i8）で済む。

### 5. 共有 `krg` の席は `karume/5` に置かない

各 `krm` がグラフ記述を内包する（グラフ JSON は 77 本で 14.2 MB — 複製の費用が小さい）。
descriptor の `graph` 文書の sha256 が同一性を与えるので、共有 `krg` を配る席（決定 20 の
`graph: FileRef`）は**要る段（段 5 の LoRA）で足す**。

### 6. hub の検査は「宣言だけで閉じるもの」に絞る

件数・天井・`size: 0` の規則・part 0 の長さ（= 24 + 2 文書の長さ）・越境の一様性・descriptor 長
（≤ 32 MiB）は parse で見る（part の長さに 64 の倍数の規則は無い — 64 B 整列は block の offset の規則で、
part 長は 4 の倍数にしかならない）。**descriptor と parts の整合・block 目次・codec 台帳の突合は
`openContainer` が持つ**（`open.ts:222-278`）ので hub には置かない — 検査点を 2 つにしない。

`resolveFiles` の戻りは **選択結果の構造型**（部品 → コンテナ参照・資産名 → FileRef）にし、
prefetch / 在庫 / 進捗が使う平坦な FileRef 列はそこから導く。`karume/4` の取得キー規約
（`<weights>[i]` / `<weights>.<extra>` — `packages/hub/src/resolve.ts:91-121`・models 側の連番復元
`packages/models/src/hub/components.ts:141-172`）は退役する。

### 7. 取得単位は**段 2 では part**。取得層は変えない

hub はコンテナ 1 本につき `BlockSource` を返す取得面を 1 本持つ。

- **cold**: 相 1（`prefetchFile` — ファイル全量を流しながら sha256 検証・記録ハッシュ焼き込み）で
  part を温め、区間読み口を開く。RAM に part は載らない。
- **warm**: 記録ハッシュの文字列比較のみ（ADR 0080 を継承）。digest は 0 回。
- **区間読みの費用型で分岐**: seek（ブラウザの Blob・ローカルの区間読み）は block ごとに読む。
  scan（Deno の既定）は part を 1 度に読んで block に切る（同じ part の block を順に読むと読み飛ばしが
  二次になる — `fetch-cache/src/core.ts:1510-1566`）。
- **「検証済み」を運ぶ**: 取得層がファイル全体を検証したバイト列（HF 経由）は `BlockSource` が
  検証済みと名乗り、`openContainer.readBlock` は**未検証の取得元にだけ** block の sha256 を掛ける
  （`fromContainer(bytes)`・ローカルディレクトリ — hub がファイル全体の sha256 を照合しない点は
  ADR 0086 決定 2 のまま）。cold の 2 重 digest（温めの逐次 + block）を避ける。container-v1 §7 の表を
  この形に訂正する。
- **HTTP Range は段 6 のまま**。前倒しの条件は「段 2 の RAM ピーク harness で cold のピークが
  『part 長 + 重ね合わせ』を超える」こと。取得層（`@hdae/fetch-cache`）は段 2 では変更しない。
- 進捗の単位は FileRef（= part）のまま。同時取得の予算は「本数と合計バイトの両方」
  （container-v1 §11）で、`karume/4` の `BYTE_BUDGET` = 1.5 GiB（ファイル全量の前確保が前提）は
  根拠が消えるので、**転送中 1 本 + 受信・検証中 1 本 = 本数 2・合計 = 選択中の最大 part 長 × 2**
  を段 2 の既定にし、2f の harness で見直す。

### 8. 段 2 の書き手は移行 CLI の**リポ丸ごとモード**

`karume migrate --manifest <karume.json> --out <dir>` が旧 `karume/4` を読み、全 (モデル, 部品,
dtype) を `krm` へ変換し、`assets` を写し、`karume/5` の `karume.json` を書く。旧 manifest から
shard 列を引くので、ディレクトリを跨ぐ shard 列（sbv2 の `shared/front` / `shared/voice`・
anima の話者別 transformer）もここで解ける。**dist.py / recipe が `krm` を直接書くのは段 3**
（container-v1 §12 の記述どおり）。

### 9. 共存期間の運用

2b が入った時点で hub は `karume/5` だけを読む。ローカルミラー 11 本は 2d で全部移行して差し替え、
レーンは移行済みミラーで回す。HF 上の pin は段 2 で irodori-v4.1-small だけ更新し、残りは段 3 の
再アップロードまで**旧版パッケージからだけ**動く（pin は 40 桁 revision なので旧版は壊れない —
ADR 0108 却下案 3）。

### 10. 部品差し替え席との関係

ADR 0108 決定 19 の `fromPretrained(source, { components: { <役割>: ComponentSource } })` は本 ADR の
`container` 欄をそのまま受ける — `ComponentSource` は「別の `karume/5` リポの (model, quant, 部品)」
またはコンテナ参照 1 本。admission は**重みを 1 バイトも取る前**に ①`graph` 文書の sha256 が
ベースの期待値と一致 ②束縛表の不足 / 余剰 0（`openContainer` の合流が descriptor だけで出す）
③席の実行ノブと codec の整合、を全件列挙で見る。型の詳細は段 2e で決める。

## Consequences

- **破壊変更**（major 繰り上げ・hub の公開面 = `resolveFiles` の戻り型と取得面の追加・
  `StreamedAsset` 系の shard 面は段 3 で退役）。CHANGELOG の Breaking は段 2 の docs 同期で書く。
- 退役する契約: shard 順序の意味（ADR 0070 決定 3・0081 決定 1）・tensor pieces（0090）・
  `SHARD_BYTE_LIMIT` の Python 突合（`shards.py:92` ↔ `manifest.ts:84`）。
- モデルカード（`modelcard.py`）と `hf-upload.zsh` の断片化表は `*.krm` を歩く形へ追随する
  （カードは段 3・upload は段 2 で前倒し）。
- 検収は ADR 0108 の段階分解表 段 2 の ①〜④。

## 追記 1 — グラフ名の規則と pin 移行の時期（2026-09-24）

段 3（3a〜3d）の実装で決めた点のうち manifest に掛かるもの。容器の形式側は ADR
[0108](0108-container-format.md) 追記 4、規則の本文は [container-v1](../container-v1.md) §2.1 が正本。

1. **容器のグラフ名 = 部品名 = `karume.json` の `weights` のキー** MUST（決定 2 の補足）。
   - **なぜ規則にするか**: 読み手はグラフを名前で引く（models の部品は `prepareContainer(opened, <weights の
     キー>)` で Session を組む）。manifest は部品 × dtype ごとに容器を 1 本指し、容器は 1 グラフだけを持つので、
     manifest から容器の中のグラフへ渡れる名前は weights のキーしか無い。書き手が別の綴りを名乗ると、
     移行済みミラー（移行 CLI のリポ丸ごとモードはキーで焼く）と再 export した系列が別物になり、しかも
     manifest の parse も容器の parse も通るので、利用者のロードで「コンテナにグラフが無い」として初めて
     落ちる。
   - **置き場のディレクトリ名は規則にしない**。系列直下に容器を置く family ではディレクトリ名が系列名に
     なり（siglip2 の系列ディレクトリに対してキーは `vision`）、irodori の `caption-proj` はキーが
     `caption_proj`、deberta の `full-24layer` はキーが `text_encoder` である。そのため export の一本道は
     `graph_name` を必須にして既定を持たず、recipe は部品名を定数で名乗る。
   - **検査点**: `karume dist` が組み立ての前に、現物の容器のグラフ名の集合とその席の weights のキーを
     突き合わせる（`DistError`）。recipe の綴りは AST の門（`tools/export-recipes/tests/test_graph_names.py`）が
     定数であることと、名乗るグラフ名が配布計画の weights のキーと対応することで固定する（例外的な family の
     扱いもこの門が持つ）。
2. **決定 9 の訂正 — 残りの pin の移行は段 3 ではなく release の波で行う**。段 3 で pin は 1 本も動いて
   いない。`karume/5` を指すのは段 2 の irodori-v4.1-small だけで、残り 9 リポは release の波まで
   旧版パッケージからだけ動く。[release-runbook](../release-runbook.md) §0 の順序（version bump → 焼き直し →
   アップロード → pin → JSR publish）が非可換 MUST で、manifest の `generator` 欄がパッケージ版を写すため、
   bump 前に上げると古い版を名乗る配布形が HF に載る。上げ直しでは新 manifest が参照しない旧ファイルを
   同じリポから消し、pin は削除後の main で焼く（HF の履歴は残す — ADR 0108 決定 18）。手順は
   release-runbook が持つ。
3. **本文の段 3 への持ち越しは済んだ**。決定 8 の「dist.py / recipe が `krm` を直接書く」は段 3a / 3b、
   Consequences の「`StreamedAsset` 系の shard 面の退役」は段 3d（hub の `streamAssets` / `StreamedAsset` /
   `StreamAssetsOptions` を削除）で入った。

## 追記 2 — 決定 7 の同時取得予算と `fromContainer(bytes)` の実名（2026-09-25）

決定 7 の末尾（段 2 の既定 = 本数 2・合計 = 最大 part 長 × 2）は採られなかった。段 2f 以降の実装は次の形で、
数え方の正本は [container-v1](../container-v1.md) §11 である。

1. **温め（相 1）は本数 4 本の streaming**（`packages/hub/src/fetch.ts` の `CONCURRENCY` — `prefetchAssets`）。
   body をそのままキャッシュへ流すので受信バッファの前確保が無く、本数からは RAM が決まらない。
2. **区間読みの保持は scan 型で part 1 本ぶんの保持枠**（`packages/hub/src/container.ts` の保持枠）。seek 型の
   保持は 0（block の区間 view だけ）。決定 7 の「本数 2」より厳しいので、実装を決定へ戻す利得は無い。
3. **`BYTE_BUDGET`（1.5 GiB）は全量面 `fetchAssets` にだけ残る**（manifest の `assets` 用 — 1 ファイルの受信
   バッファを受信前に確保する面なので、合計バイトの予算が要る）。容器の part はこの面を通らない。値は
   据え置く（2026-09-25 裁定）。`fetch.ts` の `BYTE_BUDGET` の根拠文が引く実測（anima turbo i4 の先頭 4 本で
   計 2.503 GiB の同時前確保）は `karume/4` の shard を全量面で取っていた頃のもので、今この面を通るのは資産だけ
   である。値の根拠は「1 ファイルの前確保 + 検証の一時コピーを単一 ArrayBuffer の上限より下に置く」の側に残る。
4. **`fromContainer(bytes)` の実名は `openContainer` の `{ kind: "bytes" }` 入力**
   （`packages/runtime/src/format/container/open.ts` の `ContainerInput`）。決定 7 と ADR
   [0108](0108-container-format.md)（決定 2 / 8 / 9 と追記 2 の 3）に残る `fromContainer` はこの口と読み替える。
   本文は書き換えない。
