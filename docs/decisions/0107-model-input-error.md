# 0107: 入力起因の失敗を家族横断の `ModelInputError` 1 本で表す

- Status: accepted（2026-09-22 — 型・所有者・公開面に続き、家族側の置き換えまで同日に完了）
- Date: 2026-09-22
- 対象: `packages/models/src/errors.ts`（新設 — 型）/ `packages/models/src/request-gates.ts`
  （新設 — 家族横断の受理集合の所有者）/ `packages/models/src/sbv2/errors.ts`（派生へ付け替え）/
  `packages/models/src/generation/sequence.ts`（`GenerationCapacityError` を派生へ付け替え）/
  公開面 10 面（`packages/models/mod.ts` と pipeline を出す 9 サブパス entry）・単体テスト
  `packages/models/tests/errors_test.ts`・公開面 fixture
  `packages/models/tests/fixtures/public-surface.json`。**ランタイム・hub・配布形・exporter は
  無改変**
- 関連: ADR [0008](0008-public-api.md)（公開面は薄く・増やすときは利用者ストーリーで説明する）/
  [0072](0072-sbv2-text-injection.md) 決定 6（SBV2 のどの検査が入力起因でどれが内部不変条件かの
  線引き — この ADR はその線引きを家族横断へ広げるだけで、線そのものは動かさない）/
  [0083](0083-generation-api-surface.md) 決定 10（`GenerationCapacityError` が切り詰めの実値を
  欄で運ぶ契約）/ ADR [0038](0038-manifest-v1.md) §4 系の hub `HubError`（分類の軸を借りた先）/
  [limitations](../limitations.md)（公開挙動として書いた例外の綴り）/
  [glossary](../glossary.md)

## Context

`@karume/models` は 8 家族のパイプラインを 1 パッケージで出しており、生成要求の検査は家族ごとに
書かれている。置き換え前の時点で、**入力起因**（呼び手が渡した要求そのものが受理できない = 入力を
直せば通る）の throw は 127 件あり、その型は 3 通りに割れていた:

| 型               | 件数 | どこ                                            |
| ---------------- | ---- | ----------------------------------------------- |
| 素の `Error`     | 93   | 6 家族の `generate` / `fromAssets` の入口       |
| `RangeError`     | 19   | seed の値域・sampler の指定・画像 / 音声の寸法  |
| `Sbv2InputError` | 15   | SBV2 だけ（ADR 0072 決定 6 で型を持たせた家族） |

この形だと、**複数の家族を同じホストに載せる側**（HTTP サーバー・CLI・UI）が 400 と 500 を
分ける手段が無い。素の `Error` は内部不変条件の破れ・資産（manifest / tokenizer / shard）の
齟齬・GPU 容量の不足と同じ型なので、`instanceof` では何も言えない。残るのは**メッセージの
文字列を読む**ことだけで、これは文言を変えた瞬間に黙って壊れる結合である。

`Sbv2InputError` は 1 家族でこの問題を解いているが、その解を 8 回繰り返すとホストは 8 つの型を
知ることになる。分岐先は 1 つしかないのに。

併せて、**seed の受理集合**（非負の安全整数）は anima / sbv2 / irodori の 3 家族が同じ条件を
別々に持っている。同じ値域を 3 か所に写すと、片方だけ緩む・片方だけ falsy 判定になる形の
ずれが入るのは時間の問題である。

## Decision

### 1. 型は 1 本 — `ModelInputError extends Error`

置き場は `packages/models/src/errors.ts`。`name` は `"ModelInputError"`、`cause` は
`ErrorOptions` で透過し、**追加の欄は持たない**。欄を持たせないのは、欄が要る失敗が
`GenerationCapacityError` 1 つしかなく（決定 4）、親に空の欄を置くと「いつ埋まるのか」を
呼び手が読めなくなるからである。

### 2. 分類の軸は**呼び手の分岐先**

hub の `HubError` と SBV2 の `Sbv2InputError` と同じ流儀にする。この型が飛ぶのは
「**渡した要求そのものが受理できない = 入力を直せば通る**」ときだけで、HTTP サーバーなら 400 に
当たる。次はこの型にしない（500 に当たる — 呼び手が入力を直しても直らない）:

- 内部不変条件の破れ（`sum(word2ph)` の不一致・tile 走査の破れ）
- 資産の齟齬（manifest / tokenizer / shard が宣言と食い違う）
- GPU 容量の不足

これらは素の `Error` と既存の型のまま残す。

### 3. 適用範囲は**生成要求の値域・型・組合せ**の検査だけ

次の 2 つは入力起因に見えるが、この型には混ぜない — 呼び手が打つ手が違うからである:

- **model / quant / sampler 名の綴り違い**（打つ手は「受理集合を引き直す」であって
  「値を直す」ではない）
- **呼び出し手順の違反**（`dispose` 済みの再利用・二重生成 — 打つ手は「呼ぶ順を直す」）

どちらも素の `Error` のままにする。範囲を後から**足す**のは呼び手から見て互換だが、
狭めるのは既に分岐している側を壊すので破壊変更である（決定の非対称性 — Consequences）。

### 4. 派生は 2 本だけ

- `Sbv2InputError extends ModelInputError`（`name` は `"Sbv2InputError"` のまま・ADR 0072
  決定 6 の線引きは仕様としてそのまま温存する）
- `GenerationCapacityError extends ModelInputError`（切り詰めの計算に要る実値を欄で運ぶ —
  ADR 0083 決定 10）

この 2 本は `instanceof` の外に**追加の情報**がある（家族固有の線引き / 構造化された欄）。
情報が増えない分岐先は型で割らないので、**家族ごとの専用型は作らない**。

### 5. 入力起因の `RangeError` 21 箇所も `ModelInputError` へ寄せる

対象は seed の値域・sampler の指定・画像 / 音声の寸法。数えるのは **`throw` 箇所**で、内訳は
seed 4 / anima の `steps` 1 / sampler 8 / 画像 3 / WAV 4 / 参照音声の長さ 1 である（門の本数で
数えると seed の 4 箇所は決定 6 で 1 本へ畳まれるので 18 になる — 基準が違うと数も違う）。
公開挙動の変更なので CHANGELOG の Breaking に載せる（`instanceof RangeError` で分岐していた
コードは追随が要る）。

### 6. seed の受理集合は所有者を 1 本にする

`packages/models/src/request-gates.ts` に `assertAcceptableSeed(seed: number): void` を置く
（`ModelInputError` を投げる）。受理集合は非負の安全整数で、3 家族が共有する。検査は**生成の
入口**で行う — 乱数生成器を作るのは重みを GPU に載せた後なので、生成器のコンストラクタにしか
検査が無いと、GB 級のロードを待たされた末に打ち間違いで落ちる。

## 検討した代替案

1. **標準型（`RangeError` / `TypeError`）に揃える**（新しい型を作らない）— 却下。判別子に
   ならない。`packages/models/src/` の `RangeError` は置き換え前で 54 件あり、置き換え後に残る
   33 件は**内部ヘルパの事前条件**（呼び手の入力ではなく配線の前提）なので、500 相当が同じ型に
   混ざる。`instanceof RangeError` で 400 を切り出すと、配線の破れを「入力を直せ」と呼び手に
   返すことになる。
   ADR 0072 決定 6 が「`RangeError` は分類軸の外」と決めているのはこの理由で、ここで方針を
   反転させる根拠は無い。
2. **家族ごとの専用型を 8 本作る**（`Sbv2InputError` の形を横展開する）— 却下。起票の理由を
   消せない。複数家族を載せたホストは 8 つの型を import して `instanceof` を 8 回書くか、
   結局メッセージを読むかになる。分岐先は「入力を直せ」の 1 つしかないので、型の数が増えても
   呼び手の枝は増えない。家族固有の線引きが要る 1 家族（SBV2）は決定 4 の派生で足りる。

## Consequences

- **公開面が +1**（barrel と 9 サブパスすべてから同じ 1 型が出る — 公開面 fixture の差分は
  `ModelInputError` の行だけ）。家族を 1 つしか使わない消費者も、サブパスから同じ型を掴める。
- **`RangeError` からの変更は破壊的**。seed / sampler の値域 / 画像・音声の寸法で
  `instanceof RangeError` を書いていたコードは `ModelInputError` へ書き換える。
  `ModelInputError` は `Error` の派生なので、`catch (e) { ... }` だけのコードは無影響。
- **適用範囲の拡張は足すだけ**（今は素の `Error` の検査をこの型へ移すのは、呼び手から見れば
  「500 だったものが 400 になる」= 分岐が増える方向で、既存の枝は壊れない）。**狭めるのは
  破壊変更**なので、迷ったら入れないのが安い側である。
- `GenerationCapacityError` は親が増えただけで、`name` も欄も引数の形も変わらない。
  `instanceof GenerationCapacityError` で分岐していたコードは無影響。
- **家族側の置き換えは完了している** — 生成要求の値域・型・組合せの検査 95 箇所（anima /
  generation / gemma / sbv2 / irodori / image / audio / vowel-detector）が `ModelInputError` を
  投げる（数えるのは `new ModelInputError` の箇所。初版は 73 箇所で、2026-09-22 のレビュー反映で
  取りこぼし〈gemma の投機 / PLE ノブ・irodori の参照話者・vowel-detector の波形長・PNG の
  アルファ・生成の `capacity` と空 prompt〉を足し、逆に資産の齟齬・内部不変条件だった 3 箇所
  〈anima manifest の既定解像度・anima のプロンプト上限・gemma の manifest 単独の quant 宣言〉を
  素の `Error` へ戻した）。
  メッセージは変えず型だけ差し替えた（型の分岐とメッセージの質は別物）。**例外は anima の
  `steps` 1 本**で、所有者を `sigmaSchedule` 側へ一本化したぶん入口の文言（「linspace の分母が
  0 になる」）が所有者の文言（「sigma の linspace が組めない」）へ揃った。もう 1 本は gemma の
  `estimateSessionMemory` の `capacity` 診断で、2026-09-22 のレビュー反映で整数条件（「1 以上の
  整数でない」）と大小関係（「chunkLength … を下回る」）の 2 本に割り、`createGenerationSequence` /
  `Gemma4ChatSession` と同文にした（1 本に畳んだ「未満」は `NaN` / `1.5` を大小関係として
  誤診していた）。文言が動いたのはこの 2 本だけである。
- 同じ受理集合を 2 か所に書いていた 3 本は**所有者を 1 本**にした: seed（4 本の写し →
  `request-gates.ts`）・anima の `steps`（入口と `sigmaSchedule`）・sbv2 の `styleWeight`
  （入口と表引き）。入口はいずれも所有者を呼ぶ形で、条件そのものは持たない。sbv2 の入口の
  seed 門だけは `Sbv2InputError` で包み直す（SBV2 だけを載せるホストの `instanceof` を
  壊さないため — 決定 4 の派生なので横断の分岐にも乗る）。
