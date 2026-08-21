# 0074: quant 席の命名規則 — 格納語彙 1 本 + 軸ごとのノブ

- Status: accepted（2026-08-21 — ユーザー裁定「起票お願いします」。**適用は 0.5.0 の breaking
  波**で、`linearCompute: "i8a8"` → `"a8"` の改名・[0075](0075-quant-presentation.md) の
  `karume/4` 繰り上げ・SBV2 の yomi 依存分離と同乗させる。実装は未着手）
- Date: 2026-08-21
- 関連: ADR [0041](0041-manifest-v2.md)（`presets` → `quants` の語彙）/
  [0025](0025-w8a8-linear-execution.md)・[0030](0030-attention-a8-execution.md)（`i8a8` の出自）/
  [0069](0069-packed-w4-storage.md)（i4 格納）/ [0075](0075-quant-presentation.md)（表示名・説明）

## Context

quant 席の名前はファミリごとに独立に育ち、2026-08-21 時点で 4 つのほころびがある。

| ファミリ                                    | 席                                                                | 既定       |
| ------------------------------------------- | ----------------------------------------------------------------- | ---------- |
| anima                                       | `f16` `i8` `w8a8` `w8a8-a8` `w8a8-s16` `f16-c16` `w4` `w4-a8-s16` | `w8a8-s16` |
| sbv2                                        | `f16` `w8` `w8a8` `w8-bert4` `w4`                                 | `w8-bert4` |
| irodori                                     | `f32` `f16` `w8` `w8a8`                                           | `w8a8`     |
| birefnet / depth-anything / siglip2 / vowel | `f32`                                                             | `f32`      |

1. **同じものに 2 綴り** — 「i8 格納・素の計算」が anima では `i8`、sbv2 / irodori では `w8`。
2. **同じ綴りが 2 つの軸** — `w8a8` の `a8` は linear の活性、`w8a8-a8` の 2 つめの `-a8` は
   attention の活性。2026-08-21 に新設した `w4-a8-s16` もこの曖昧さを引き継いでいる。
3. **名前が実態と食い違う** — sbv2 の `f16` 席は純粋な f16 ではなく `text_encoder` だけ i8
   （`sbv2/distribution.py`）。
4. **文法の違う軸が混在** — `w8-bert4` は「部品ごとの上書き」で、他の席名と組み立て方が別。

hub は quant 名を**不透明なキーとしてしか扱わない**（`resolveFiles` は
`Object.hasOwn(entry.quants, quantName)` を見るだけ）。したがって改名はデータと文書の変更で、
パーサや runtime の工事を伴わない。

## Decision

### 1. 文法

    <格納>[+<部品><ビット>]…[-<ノブ>]…

- `<格納>` = 全役割に共通の**基底**格納
- `+<部品><ビット>` = 役割ごとの上書き（0 個以上）
- `-<ノブ>` = 実行時のノブ（0 個以上・下の定義順に並べる）

### 2. 格納の語彙は資産ヘッダと 1:1（`f32` / `f16` / `i8` / `i4`）

`w8` / `w4` という第 2 語彙を廃す。席の `weights` 宣言に書く dtype ラベル・
`STORAGE_REQUIREMENTS` が要求するヘッダ dtype・席名の 3 つが同じ綴りになり、
「どれが正か」を考える場面が消える。`i4` は混成（適格な重みが i4・残りが i8）を指すラベルで、
これは既に `ANIMA_WEIGHTS` / `SBV2` の席が使っている意味そのもの。

### 3. ノブの綴り — `a8` / `attn8` / `s16` / `c16`

| トークン | 軸                               | session の値                                |
| -------- | -------------------------------- | ------------------------------------------- |
| `a8`     | linear の活性を i8 へ            | `linearCompute: "a8"`                       |
| `attn8`  | attention の活性を i8 へ         | `attentionCompute: "a8"`                    |
| `s16`    | attention の score 格納を f16 へ | `attentionScoreStorage: "f16"`              |
| `c16`    | 計算を f16 へ（shaderF16 要求）  | `linearCompute` / `attentionCompute: "f16"` |

**値の語彙（`a8`）と席名のトークン（`a8` / `attn8`）は別レイヤ** — オプションのキーが既に軸を
名指ししているので値は軸を持たない（`attentionCompute: "a8"`）。席名は 1 本の文字列に軸を
畳むので、どちらの活性かをトークンで分ける。

`linearCompute` の値を `i8a8` → `a8` にするのは、このノブが決めているのが**活性の扱いだけ**だから
（重みの格納は資産ヘッダが決める）。旧綴りの前半 `i8` は重み格納の判別子 `:wi8` を畳んだもので
（[dp4a-w8a8-design](../research/2026-08-03-dp4a-w8a8-design.md) — 逐語「`v2` の重み格納判別子 `:wi8` は i8a8 では意味が重複するので `i8a8` に畳む」）、i4 常駐の重みを同じ経路へ
載せる w4a8 が入ると事実として嘘になる。**格納形を値に書くと、資産ヘッダと manifest の二重持ちに
なる**（CLAUDE.md の「派生状態を独立更新の非正規化フィールドに持たない」）ため、値を格納形ごとに
増やす案（`i4a8` を足す）は採らない — 混成資産（anima の i4 系列は i4 453 本 + i8 1 本）で
どちらか一方しか加速できなくなる実害もある。

### 4. 部品上書きの略称は recipe が定め、カードで対応を明示する

`i8+bert4` の `bert` のような略称は役割名（`text_encoder`）より短く読みやすい一方、
それ自体が語彙になる。**略称の定義は recipe（`distribution.py`）が持ち、生成モデルカードの
quant 表に対応を必ず出す**。[0075](0075-quant-presentation.md) の表示名・説明が入れば UI 側では
略称を読ませずに済む。

### 5. 既定は名前で表さない

「推奨」「default」を席名に含めない — `defaultQuant` が既に指しているので、名前に書くと
二重持ちになる（同じ理由で [0075](0075-quant-presentation.md) も `recommended` 欄を持たない）。

### 6. 移行表（0.5.0 で一斉に）

| 現行                                  | 0.5.0             | 備考                                     |
| ------------------------------------- | ----------------- | ---------------------------------------- |
| anima `i8` / sbv2 `w8` / irodori `w8` | `i8`              | 2 綴りの統合                             |
| `w8a8`                                | `i8-a8`           |                                          |
| anima `w8a8-a8`                       | `i8-a8-attn8`     |                                          |
| anima `w8a8-s16`（既定）              | `i8-a8-attn8-s16` |                                          |
| anima `w4-a8-s16`                     | `i4-attn8-s16`    | 現行名の `-a8` は attention の意味だった |
| sbv2 / anima `w4`                     | `i4`              |                                          |
| sbv2 `w8-bert4`（既定）               | `i8+bert4`        |                                          |
| sbv2 `f16`                            | `f16+bert8`       | 実態（text_encoder だけ i8）に合わせる   |
| anima `f16-c16`                       | `f16-c16`         | 変化なし                                 |
| `f32`                                 | `f32`             | 変化なし                                 |

新しい名前は長くなる（`i8-a8-attn8-s16`）が、**id は機械の都合で、人が読むのは
[0075](0075-quant-presentation.md) の `label` / `description`** という役割分担を取る。

## Consequences

- 公開済み 3 リポ（jvnv / irodori / anima）の manifest が全て変わる → 再アップロード + pin 更新
  （[0073](0073-models-source-pin.md)）が要る。0.5.0 の他の breaking と 1 回にまとめる。
- hub / runtime のコード変更は**不要**（名前は不透明キー）。変わるのは recipe の quant 表・
  examples・E2E テストの参照名・docs。`linearCompute` の値の改名だけは runtime の型と hub の
  allowlist に触れる（[0075](0075-quant-presentation.md) の format 繰り上げと同じ波）。
- 旧名の互換シムは**作らない**。リリース済みだが、breaking 波でまとめて切り替える方が、
  「どちらの綴りも通る」期間を作るより誤解が少ない（旧名で pin された 0.4.0 クライアントは
  旧 revision を読み続けるので壊れない — pin が緩衝材になる）。
- 小ファミリ（`f32` のみ）は無変更。
