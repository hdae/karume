# 0092: 配布リポの粒度・命名と、models が持つ公開リポ対応表

- Status: accepted（2026-09-04 — ユーザー裁定。波 b〈未配布家族の初回公開〉の前提として
  リポ割りの規則を先に固める）
- Date: 2026-09-04
- 関連: ADR [0038](0038-manifest-v1.md) §7（越境コンポーネント参照 — 参照先の revision を焼く）/
  [0041](0041-manifest-v2.md)（リポ名の `karume-` prefix・1 リポ複数モデル + `defaultModel`）/
  [0065](0065-exporter-core-recipe-split.md)（core = PyPI / recipe = repo-only の分離）/
  [0073](0073-models-source-pin.md)（pin の必要性 — **定数の形は本 ADR で置換**）/
  [0087](0087-anima-official-extra-repos.md)（anima の「公式 / 追加学習」軸 = 粒度規則の先例）

## Context

**リポ割りが都度裁定のまま溜まった。** 既存 6 リポの割り方（`karume-anima` /
`karume-anima-extra` / `karume-sbv2-jvnv` / `karume-irodori-v4-small` /
`karume-irodori-v4.1-small` / `karume-gemma4-e2b`）はそれぞれ個別の裁定で決めており、
規則としてはどこにも書かれていない。波 b では未配布の家族（siglip2 の base / so400m、
birefnet の BiRefNet HR / Lucida、depth-anything v2、vowel-detector）を初めて公開するが、
どれも「同居か分離か」がまず問われる形をしていて、規則がないと再び 1 件ずつ裁定になる。

**pin 定数の形が「既定 1 つ」時代の残骸になっている。** ADR 0073 は当初、決定 2 で
`fromPretrained` の `ref` を optional にし、家族ごとに既定ソース 1 つを焼く設計だった。
2026-08-25 にその決定 2 は撤回され（既定席そのものを廃止）、定数は
`<FAMILY>[_<VARIANT>]_CURRENT` へ改名して「パッケージ版が検証した取得元」という位置づけに
変わった。しかし**形は「1 公開リポ = 1 定数」のまま**で、家族にリポが増えるたびに
トップレベル定数が増える（現状 6 本）。「この家族の公開リポの一覧」を表す席が型に無く、
新リポの追加は「定数を 1 本足して barrel の re-export も足す」という**足し忘れうる手作業**に
なっている。違和感の正体はこれで、pin という決定が悪いのではなく**入れ物が既定 1 つ時代の
形のまま**である。

**revision の在処は TS 側にしか置けない。** HF のモデルカードは自分自身の revision を書けない
（カードを書く時点でその commit はまだ存在しない）。docs へ SHA を写さない規約
（release-runbook §3）はこの帰結で、写しは焼き直しのたびに古びる。したがって
「公開リポ → その版」の対応は **models の定数 1 か所**が唯一の在処になる。

**recipes 分離の動機はライセンスであって構造ではない。** backlog parked
「export-recipes の別リポジトリ分離」は、上流由来コード（patch 層・参照 pipeline・export 台本）が
本リポの MIT の下にあるように読めることへの懸念が出発点だった。構造の分離は ADR 0065 が
machine gate まで含めて既に済ませており、別リポ化が追加で買うのは「見え方」だけである。
一方で払う代償は uv workspace の解体・資産根と fixture 書き先の注入・両側 CI で、実務コストは
小さくない。

**ライセンス的に面倒な派生の受け皿も要る。** 学習データが不明なもの、検閲解除派生など、
再配布の書面根拠を本プロジェクトが引き受けたくない資産がある（`karume-sbv2-fn` の保留が先例）。
これを models のパイプラインや対応表に載せると、説明責任がプロジェクト本体へ掛かる。

## Decision

### 1. 粒度 = 家族 1 リポ。派生は別リポ、世代・版も別リポ

- **同一家族の変種は 1 リポに同居**する（Gemma 4 の E2B / E4B / 12B は
  `karume-gemma4` に同居）。ADR 0041 の「1 リポ複数モデル + `defaultModel`」がそのまま器。
- **派生モデルは別リポ**（BiRefNet に対する Lucida は `karume-lucida`）。
- **派生が多い家族は `-extra` の束**でまとめる（anima — ADR 0087 の先例をそのまま規則化）。
- **世代・版が変わるものは別リポ**（irodori v4 と v4.1 は別・Gemma 5 が出たら新規リポ）。
- **BiRefNet と BiRefNet HR は別**（同名家族だが別モデルとして配る）。

### 2. リポ命名

- 公式: `karume-<family>[-<世代 or 版>][-<変種>]`
  （`karume-gemma4` / `karume-irodori-v4.1-small` / `karume-birefnet-hr` /
  `karume-depth-anything-v2` / `karume-siglip2`）
- 派生の束: `karume-<family>-extra`
- 単独の派生: `karume-<派生名>`（`karume-lucida`）
- **版のドットは容認する**（`karume-irodori-v4.1-small` — 上流の名乗りを崩さない方を採る。
  ADR [0077](0077-model-version-naming.md) の動機と同じ）。

`karume-` prefix は ADR 0041 の裁定を継承（HF org を作らずリポ名で名前空間を切る）。

### 3. models の公開リポ対応表 = `<FAMILY>_SOURCES`

家族ごとに、その家族の**公開リポ全部**を 1 つの表で持つ。

```ts
export const IRODORI_SOURCES = {
  "irodori-v4-small": { repo: "hdae/karume-irodori-v4-small", revision: "<40hex>" },
  "irodori-v4.1-small": { repo: "hdae/karume-irodori-v4.1-small", revision: "<40hex>" },
} as const satisfies Record<string, HubRepoRef>;
```

- **キー = HF リポ名の basename から `karume-` を落としたもの**。ハイフン・ドットをそのまま
  含むのでブラケットアクセスになる（`IRODORI_SOURCES["irodori-v4.1-small"]`）。識別子として
  綴りやすい別名を発明すると、リポ名との対応が人間の記憶に落ちる。
- **単一リポの家族もキーは家族名**（`GEMMA4_SOURCES["gemma4"]`）— 1 本でも表の形を崩さない。
  リポが増えたときに形が変わらないことが、この決定の目的そのもの。
- **barrel には全家族を畳んだ `KARUME_SOURCES` を置く**（キーは同じ）。「公開リポの一覧」が
  1 か所で読める席で、疎通テスト（`tools/published-smoke`）と pin 更新の網羅検査はここを回す。
- **既定席は作らない**（ADR 0073 の 2026-08-25 撤回を維持 — 取得元は呼び出し側が必ず綴る。
  ADR [0086](0086-distribution-source.md) 冒頭の「`ref` 必須の MUST を取得元ハンドルへ継承」も
  同じ向き）。
- **値の revision は公開時点の commit SHA を焼き、bump のたびに更新する**
  （ADR 0073 決定 3 の手書き + 手順書ゲート・維持義務ともに継承）。
- **不変条件を機械で門にする**: `"karume-" + key === repo の basename`・owner は `hdae`・
  `revision` は 40 桁の hex。

### 4. `*_CURRENT` は廃止する（breaking・0.9.0）

`<FAMILY>[_<VARIANT>]_CURRENT`（6 本）は決定 3 の表へ移す。未リリース側の JSR 公開面の
breaking 変更で、互換シムは置かない（0.9.0 — CLAUDE.md の未リリース方針）。

### 5. export-recipes は本リポに残す（分離動機は carve-out で解く）

ADR 0065 決定 1 の repo-only は不変で、**別リポ化はしない**。ライセンス面の懸念は文章で切る:

- リポ直下 README の License 節と `tools/export-recipes/` の README 冒頭に、
  **「この下は上流由来のコードを含み、家族ごとの `THIRD_PARTY_NOTICES.md` に従う。
  プロジェクトの MIT の対象外」**を明記する。
- **MIT 由来の逐語移植を自己申告する家族**（birefnet の `patch.py` / `export.py`）は、
  家族の `THIRD_PARTY_NOTICES.md` に**上流の著作権表示そのもの**を載せる（MIT §「著作権表示と
  許諾表示を含めること」の履行）。

backlog parked の「export-recipes の別リポジトリ分離」は本決定でクローズする。

### 6. ライセンス的に面倒な派生は本リポに入れない

学習データが不明なもの・検閲解除派生など、再配布の書面根拠をプロジェクトが引き受けられない
資産は、**exporter core を使う別ライブラリ + 別 HF リポ**として組む。models にパイプラインを
足さず、決定 3 の対応表にも載せない。PyPI `karume` は未リリースなので、その別ライブラリは
当面 path 依存で core を参照する。

### 7. `LICENSE.md` / `NOTICE.md` を配布リポ直下に同梱する

配布リポ直下に置けるのは `LICENSE.md` / `NOTICE.md` だけ（`verify_dist` の `LEGAL_PATHS`）。
上流ライセンスに応じて:

- **Apache-2.0**: 全文 + §4(b) の改変告知
- **MIT**: 全文 + 著作権行

今回の対象は siglip2 / birefnet-hr / lucida / depth-anything-v2。**既公開の
`karume-irodori-*`（MIT）と `karume-sbv2-jvnv`（CC BY-SA）は同梱が漏れている** — 次に上げ直す
回で是正する（backlog へ起票）。

### 8. siglip2 は 1 リポ 2 モデル（既定 = base）

`karume-siglip2` に base と so400m を同居させる（決定 1 の家族 1 リポがそのまま掛かる）。
これに伴い、siglip2 recipe の「1 モデル 1 リポ」制約は**撤回**する。

## Consequences

- **0.9.0 の breaking**: `*_CURRENT` 6 本が消え `<FAMILY>_SOURCES` + `KARUME_SOURCES` になる。
  examples / `tools/published-smoke` / モデルカードの Usage 例が追随する。
- **runbook §3 の手順が変わる**: 「定数を 1 本探して書く」から「該当キーへ記入 →
  `KARUME_SOURCES` の網羅を確認 → 疎通」へ。初公開リポの**エントリ新設もこの時点**
  （公開前に置くと 404 にしかならないキーが公開面に生える — ADR 0073 決定 1 の理由を継承）。
- **`karume-gemma4-e2b` は `karume-gemma4` へ改名する**（決定 1 — E4B / 12B が同居する器に
  なるため）。HF の rename で旧名はリダイレクトされる。改名操作は次リリースの手順（runbook §2）
  に置く。
- **波 b の対象が確定する**: siglip2（1 リポ 2 モデル）/ birefnet-hr・lucida（**2048² 対応の
  実現性を確認してから** — 前回の export は 1024² のみ）/ depth-anything-v2。
  **vowel-detector は波から外す**（上流の体裁整備が先 — 2026-09-04 ユーザー裁定）。
- **docs に SHA を写さない規約は不変**（runbook §3 の検査行も不変）。対応表の値だけが在処である、
  という性質は定数から表へ移っても変わらない。
- **ADR [0050](0050-irodori-quant-series.md) / [0073](0073-models-source-pin.md) /
  [0087](0087-anima-official-extra-repos.md) に残る `*_CURRENT` の綴りは当時の記録で、
  公開面には存在しない**（置換は決定 3）。
- 決定 5 により uv workspace（`tools/` = ルート・exporter と export-recipes の 2 member）は
  そのまま。両側の `uv run pytest` を回す検証手順も不変。
