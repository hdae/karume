# 0075: quant プリセットの表示名と説明 — アプリの選択 UI へ出す欄

- Status: accepted（2026-08-21 — ユーザー提案「量子化の選択肢を表示するアプリの場合のために、
  プリセットに表示名と説明とかを書けるようにしたら良いかもしれません」→ 起票裁定。**適用は
  0.5.0 の breaking 波**で [0074](0074-quant-seat-naming.md) の改名と同乗。実装は未着手）
- Date: 2026-08-21
- 関連: ADR [0038](0038-manifest-v1.md)（format 繰り上げ規則・session allowlist）/
  [0041](0041-manifest-v2.md)（`quants` の語彙）/ [0071](0071-manifest-v3-shards.md)（`karume/3`）/
  [0074](0074-quant-seat-naming.md)（席名の規則 — 本 ADR と役割を分ける）

## Context

quant 席は今のところ**機械名だけ**を持つ（`w8a8-s16` / `w8-bert4`）。この名前は
「どの格納・どのノブか」を畳んだ識別子で、選択 UI を持つアプリ（例: 量子化を利用者に選ばせる
デモやツール）がそのまま画面に出すには不親切であり、[0074](0074-quant-seat-naming.md) の
規則化で機械名は**さらに長くなる**（`i8-a8-attn8-s16`）。

一方で「この席は何が嬉しいのか」を知っているのは recipe を書いた人間で、その知識は今
モデルカードの散文にしか残らない。カードは README であってアプリからは読めない。

**互換性の制約（実コードで確認済み）**: manifest パーサは quant エントリに厳格な allowlist を
持ち、未知キーは fail loudly する（`QUANT_KEYS = ["weights", "session", "gpuFeatures"]` を
`assertAllowedKeys` で検査 — `packages/hub/src/manifest.ts`）。したがって欄の追加は
**optional であっても後方互換ではない** — 旧クライアントは新しい manifest を読めない。
これは ADR [0038](0038-manifest-v1.md) 決定 1 と**矛盾しない** — 同 ADR は「エンベロープ内の
未知キーは fail loudly で拒否する。additive 進化が許されるのは `pipelineConfig` の内側と §7 の
明示列挙席だけ」と定め、§7 の見出し自身が「ここに無い追加は `karume/2`」と言っている。表示欄は
その列挙に無いので、major 繰り上げは**0038 が最初から指定していた手順そのもの**である。

NOTE: 本当の食い違いは別の位置にある — §7 が「据え置きで足せる」と列挙した 2 席（ファイル参照の
`repo` / `revision`・preset の `requiredLimits`）は現行の `FILE_REF_KEYS` / `QUANT_KEYS` に席が無く、
今足せばやはり旧クライアントが読めない。据え置きの約束が実装と食い違っている。

## Decision

### 1. quant エントリに `label` と `description` を optional で足す

```jsonc
"i8-a8-attn8-s16": {
  "weights": { /* … */ },
  "session": { /* … */ },
  "label": "Balanced (int8)",
  "description": "Half the download of f16 with no visible difference."
}
```

- `label` = 選択肢に出す短い表示名（**64 字上限**）
- `description` = 1 行の説明（**200 字上限**）
- どちらも**英語**（配布物の README / モデルカードと同じ規約）。多言語欄は持たない —
  アプリ側が席 id をキーに自前の訳を当てられる
- 未設定でも動く（optional）。設定が無い席は呼び手が id をそのまま出す

### 2. hub は上限を検査するだけで、意味を解釈しない

長さ上限は「manifest は外部入力」という前提から来る境界検査で、内容の妥当性
（説明が実態と合っているか）は検査できないし、しない。表示は完全に呼び手の責任。

### 3. `recommended` / 並び順の欄は持たない

- 「推奨」は `defaultQuant` が既に指している
- 並び順はアプリが**取得量**で概ね決められる。ただし取得量は全順序を与えない — **計算ノブ
  （`linearCompute` / `attentionCompute`）だけが違う席は同じ重みを指すので取得量が同点になる**
  （anima の 8 席は 3 値へ潰れる）。同点の中は `quants` の**宣言順**（JSON の挿入順）で並べる
  のが一次キー。
- 取得量の算出は `resolveFiles` の戻り値から出すが、**「`weights` から解決したファイルの合計」
  とは一致しない** — 戻り値は `assets`（quant 非依存）も含み、複数キーが同じ `path` を指し得る
  （取得層は path で一意化してから合計する）。数えるときは同じ一意化を通すこと。

`recommended` / 並び順の欄を持たないのは、導出できるものを独立更新の欄で持つと食い違ったとき
に正が無くなるため（CLAUDE.md の「派生状態を独立更新の非正規化フィールドに持たない」）。

### 4. format を `karume/4` へ繰り上げる

Context の allowlist の性質上、欄を足した manifest は旧クライアントから読めない。
**黙って読めない形にせず、format major で断絶を宣言する**（ADR 0041 §1 の「未知 major は
fail loudly」に乗る）。0.5.0 の breaking 波にまとめ、公開 4 リポの再アップロードと pin 更新
（[0073](0073-models-source-pin.md)）を 1 回で済ませる。

### 5. モデルカードの quant 表も同じ文字列から出す

説明の綴りが manifest とカードで独立に育つと、片方だけ古くなる。カード生成
（`karume.modelcard`）は manifest 由来の値だけを印字する MUST が既にあるので、その規律に乗せる。

recipe が既に持っている英語散文（`*_NOTICE_MARKDOWN` 等の改変告知）とは**責務が別**で、
統合も相互参照もしない: 告知は「このリポの重みに何をしたか」を Pipeline へ固定で載せる 1 組で、
`description` は「席 1 つが何か」を席ごとに書く。告知側へ席の説明を足すと、席の増減で告知が
中身と食い違う（同じ理由で告知はモデルの並びにも依存させない）。

## Consequences

- `karume/4` の繰り上げは [0074](0074-quant-seat-naming.md) の席名改名・`linearCompute` の値の
  改名と**同じ波でしか出さない**（format 断絶を 2 回起こさない）。
- hub は `QUANT_KEYS` に 2 キーと長さ検査を足すだけ。resolve 側は無変更（表示欄は解決に効かない）。
- recipe は席ごとに 2 行の英語を書く義務が増える。**書かない席があってよい**（optional）ので、
  小ファミリの `f32` 単席は据え置ける。
- 席名が長くなること（[0074](0074-quant-seat-naming.md)）の代償を UI 側で吸収できる —
  この 2 本の ADR は対で意味を持つ。
