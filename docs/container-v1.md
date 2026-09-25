# Karume コンテナ v1 仕様

ADR [0108](decisions/0108-container-format.md) の具体化。`krm`（モデル）と `krg`（グラフ）の
**物理形式・descriptor スキーマ・束縛表・codec 台帳**の正本。

この文書が持たないもの:

- **IR v2 のグラフ表現そのもの** → [ir-v2.md](ir-v2.md) が正本（v1 からの差分は §13・正準直列化の
  規則も ir-v2.md の「正準直列化」節）
- **manifest `karume/5`** → ADR [0109](decisions/0109-manifest-v5-container.md) が正本
- **op の契約** → 実装の契約テーブル（`packages/runtime/src/ops/contracts.ts`）が正本
- **実行既定**（`session` / `gpuFeatures` / `requiredLimits` / `label` / `description`）→
  manifest の所有（ADR 0038 §3）。descriptor は持たない

改訂履歴（0.x の間は両読みもシムも作らない — 旧形式は移行 CLI で変換する〈§12〉。
根拠は ADR [0108](decisions/0108-container-format.md) の Consequences と決定 18〈版の分岐〉・改訂の手順は ADR 0003）:

- v1（2026-09-22）: 初版。ADR 0108 の決定 1〜20 を具体化。
- v1 訂正（2026-09-22・段 1 の実装で判明）: ①`capabilities.codecs` を**モデル記述の `codecs`** へ移す
  （グラフ記述は `krm` / `krg` でバイト同一 MUST なので、束縛表に依存する欄を持てない）②束縛表から
  借用形 `{ "shared": true }` を外す（shared 宣言は IR 側だけが持ち、束縛表のキー集合は
  「shared でも const 供給でもない initializer」と完全一致）③`rowAxis` / `groupSize` / `scale` は
  量子化 codec でのみ書く（非量子化で書くと fail loudly）④`int8-sym` の packing を 1 要素 / 1 バイト /
  整列 4 に訂正（4 / 4 / 1 だと `numel % 4 == 0` という v1 に無い制約が入る）⑤descriptor の配列順を
  書き手の決定性のために固定（`const.blocks` は offset 昇順 MUST・`blocks` は (part, offset) 昇順・
  `constants` は (graph, initializer) の code point 順・数値と map の綴りは ir-v2.md の正準直列化と同じ）。
- v1 訂正 2（2026-09-22・段 2 の裁定 — ADR 0109 / ADR 0108 追記 2）: ①§7 の digest 表を取得元の
  検証済みの有無で分ける（HF 経由は取得層の 1 回のみ・block の digest は未検証の取得元だけ）②§8 に
  manifest 側の形（`container.descriptor` = 2 文書の期待値・`container.parts` = part 0 を含む全 part・
  長さ 0 の part は 0 バイトのファイル・共有 `krg` の席は置かない）③§11 に段 2 の取得単位（part・
  seek / scan の分岐）④§12 に移行 CLI のリポ丸ごとモード（`karume/5` を書く）⑤§2.2 の `assets` に
  `length`（payload の論理長・block 長はその 4 の倍数への切り上げ MUST）を足し、PLE の役割名を実装の
  `ple-values` / `ple-scales` / `ple-index` に揃える。
- v1 訂正 3（2026-09-24・段 3 の実装 — ADR 0108 追記 4 / 5・ADR 0109 追記 1）: ①§2.1 に**グラフ名の
  規則**（容器のグラフ名 = 部品名 = `weights` のキー MUST）②§4.2 の part 長の既定は 256 MiB のまま・
  §4.1 の block 上限の見直しは段 6 へ ③§11 の RAM の数え方を 3 項（呼び手の持ち物・処理中の item・
  取得の保持）にし、持ち越し scale の写しを足す・フェンスは part ごと 1 回で staging だけを律する
  ④§12 を現行の CLI 面へ（2 モード・`provenance.writer` は既定で書かない・`--part-bytes`・
  `--cross-repo`）⑤削除済みの実装への参照を現行の置き場へ。
- v1 訂正 4（2026-09-25・実装との照合）: ①§1 の「モデル記述の直後を 64 B 整列まで詰め、それが
  part 0 の末尾」を「part 0 = ヘッダ + 2 文書ちょうど（詰め物なし）」へ訂正する。§3 / §8 の「長さ 0 の
  part の前に詰め物を挿まない」MUST と矛盾していた。3 実装（runtime の `derivePartOffsets`・exporter の
  鏡像・hub の part 0 長の照合）はもとから訂正後の規則で書いている ②§8 の表の part 0 / part 1 欄を同じ
  意味へ ③§9 の式に `[詰め物]` を足し、詰め物は写さず導き直すことを明記する。

## 0. 記法と共通規則

- 多バイト整数はすべて**リトルエンディアン**。`u32` / `u64` は符号なし。
- JSON は UTF-8。**`NaN` / `Infinity` リテラルは禁止**（書き手は `allow_nan=False` 相当で書き、
  読み手は検出したら fail loudly）。IR v1 から継承。
- **未知のキーは fail loudly**（前方互換チャネルを持たない — IR v1 から継承）。本書が
  列挙したキーが全てで、省略可能と明記した欄だけが省略できる。
- 「`sha256`」欄は常に**小文字 16 進 64 文字**の文字列。
- 「block id」は 1〜64 文字の文字列（`[A-Za-z0-9._:-]+`）。
- shape の要素は非負整数、または `coeff·sym+offset` の正準表記（IR v1 の次元言語と同一）。
- 未対応・宣言外・不足・余剰はすべて**全件列挙で拒否**する（黙って近似しない）。
- JSON のキー `__proto__` は**書けない・読めない**（読み手は非有限数・深さと同じ門で拒否し、書き手は
  正準直列化で拒否する）。素の `{}` へ代入すると [[Prototype]] 設定に化けて宣言が黙って消える名前で、
  テンソルキーがそのまま initializer 名になる v2 では理屈の上で綴れてしまうため、両側で塞ぐ。
- 本書の JSON 例に現れる `"…"` は**紙面上の省略**である（実物は §0 の規則どおり小文字 16 進
  64 文字）。例は形を示すもので、値そのものは規範ではない — 規範は各節の欄の表である。

## 1. ヘッダ

固定長 **24 バイト**。

| offset | 長さ | 欄                            | 値                                              |
| -----: | ---: | ----------------------------- | ----------------------------------------------- |
|      0 |    4 | magic                         | `KRMC`（`krm`）/ `KRGC`（`krg`）の ASCII 4 文字 |
|      4 |    4 | `u32` container version       | `1`                                             |
|      8 |    8 | `u64` graph descriptor length | 1 以上・§10 の上限以下                          |
|     16 |    8 | `u64` model descriptor length | `krm` では 1 以上・**`krg` では `0` MUST**      |

- 種別（`krm` / `krg`）を持つのは **magic だけ**。descriptor の中に `kind` 欄は置かない
  （同じ事実を 2 か所に持たせない）。
- magic が `KRGC` なのに model descriptor length が 0 でない、またはその逆は fail loudly。
- ヘッダ直後にグラフ記述、その直後にモデル記述が**詰めて**続く（両者の間に padding は無い）。
- **part 0 はヘッダ + 2 文書ちょうど**で、詰め物を含まない（分割形の part 0 のファイル長 =
  24 + グラフ記述長 + モデル記述長）。単一形の part 間の詰め物（0x00）は §8 の配置規則が決める —
  長さ 0 でない次の part の先頭が 64 B 整列になるよう、その直前にだけ詰める（例: part 0 が 100 B・
  part 1 が 0 B・part 2 が 30 B なら、part offset は 0 / 100 / 128）。

## 2. descriptor（2 文書）

descriptor を**グラフ記述**と**モデル記述**の 2 文書に割る理由は、`krg` をバイトコピーで
抜けるようにするため（§9）。

### 2.1 グラフ記述

```jsonc
{
  "format": "karume-container",
  "version": 1,
  "capabilities": {
    "ops": ["matmul", "linear", "..."],
    "features": []
  },
  "graphs": {
    "transformer": {/* IR v2 グラフ（§13） */}
  },
  "const": {
    "length": 30736384,
    "blocks": [
      { "id": "const.rope.inv_freq", "offset": 0, "length": 512, "sha256": "…" }
    ],
    "constants": [
      {
        "graph": "transformer",
        "initializer": "rope_inv_freq",
        "block": "const.rope.inv_freq",
        "encoding": {/* §6.1 */}
      }
    ]
  }
}
```

| 欄                      | 型       | 必須 | 上限・規則                                                                                                 |
| ----------------------- | -------- | ---- | ---------------------------------------------------------------------------------------------------------- |
| `format`                | string   | 必須 | 固定 `"karume-container"`。不一致は fail loudly                                                            |
| `version`               | integer  | 必須 | 固定 `1`                                                                                                   |
| `capabilities.ops`      | string[] | 必須 | 全 `graphs` の `requires.ops` の和集合と**完全一致**。読み手は自分の対応表と突合し非対応 op を列挙して拒否 |
| `capabilities.features` | string[] | 必須 | 将来の拡張点。初版は空配列のみ受理（非空は fail loudly）                                                   |
| `graphs`                | object   | 必須 | 1 個以上・**64 個以下**。キーはグラフ名（1〜64 文字・`[A-Za-z0-9._-]+`）。値は IR v2 グラフ                |
| `const.length`          | u64      | 必須 | const 領域のバイト長（§3）。const が無いときは `0`                                                         |
| `const.blocks`          | 配列     | 必須 | const block の目次（§3）。const が無いときは空配列                                                         |
| `const.constants`       | 配列     | 必須 | const block を initializer へ結ぶ**束縛表**（§3）。const が無いときは空配列                                |

**グラフ名の規則**（配布形を書く経路すべてに掛かる — export の一本道・`karume dist`・`karume migrate`）:

- **容器のグラフ名 = 配布形の部品名 = `karume.json` の `weights` のキー** MUST。ランタイムはグラフを
  名前で引く（`prepareContainer(opened, <weights のキー>)`）ので、綴りが割れた容器は manifest ごと据わり、
  利用者の `createSession` で初めて「コンテナにグラフが無い」で落ちる（理由の記録は ADR
  [0109](decisions/0109-manifest-v5-container.md) 追記 1）。
- **置き場のディレクトリ名は規則ではない**。部品名と一致しないことがある（系列直下に容器を置く family
  ではディレクトリ名が系列名になる — siglip2 のキーは `vision`。irodori の `caption-proj` ディレクトリの
  キーは `caption_proj`）。書き手は部品名を定数で名乗り、ディレクトリから導かない（export の一本道
  `karume.pipeline` は `graph_name` を必須にして既定を持たない）。
- 検査は 2 箇所: `karume dist` が組み立ての前に、現物の容器のグラフ名の集合がその席の `weights` の
  キーと一致することを見る（`tools/exporter/src/karume/dist.py` — 食い違いは `DistError`）。models の
  合流（`packages/models/src/hub/components.ts`）は `weights` のキーでグラフを引き、無ければ
  `prepareContainer` が在るグラフを列挙して落ちる。

`const.blocks[]` の要素:

| 欄       | 型     | 必須 | 規則                                                         |
| -------- | ------ | ---- | ------------------------------------------------------------ |
| `id`     | string | 必須 | block id（§0）。モデル記述の `blocks[].id` と**素集合** MUST |
| `offset` | u64    | 必須 | **const 領域の先頭からの相対** offset。64 の倍数 MUST        |
| `length` | u64    | 必須 | 4 の倍数 MUST・§10 の block 上限以下                         |
| `sha256` | string | 必須 | この block のバイト列（padding 含む）の sha256               |

`const.blocks` は `offset` 昇順で並び、互いに重ならず、`const.length` を超えない。

`const.constants[]` の要素:

| 欄            | 型     | 必須 | 規則                                                            |
| ------------- | ------ | ---- | --------------------------------------------------------------- |
| `graph`       | string | 必須 | `graphs` のキーの 1 つ                                          |
| `initializer` | string | 必須 | そのグラフの initializer 名。`(graph, initializer)` は一意 MUST |
| `block`       | string | 必須 | `const.blocks[].id` を指す                                      |
| `encoding`    | object | 必須 | §6.1                                                            |

**`const` の束縛表がモデル記述ではなくグラフ記述にあるのは、`krg` 単独で `const.*` を供給できる
唯一の手段だからである**（`krg` はモデル記述を持たない — §9）。
`const.length` も同じ理由でグラフ記述が持つ（`krg` が part 1 の寸法を知る唯一の手段）。
CPU 試作 ① の綴りは `constRegion.constants[]` / `constRegion.length` だった — 本書は
グラフ記述の `const` オブジェクトに畳んでいる。

### 2.2 モデル記述

`krg` では存在しない（ヘッダの model descriptor length が 0）。

```jsonc
{
  "format": "karume-model",
  "version": 1,
  "codecs": ["int4-sym-g"],
  "parts": [
    { "index": 1, "length": 30801920, "sha256": "…" },
    { "index": 2, "length": 268435456, "sha256": "…" }
  ],
  "blocks": [
    { "id": "w.0", "part": 2, "offset": 0, "length": 33554432, "sha256": "…", "role": "weight" },
    { "id": "s.0", "part": 2, "offset": 33554432, "length": 6144, "sha256": "…", "role": "scale" }
  ],
  "binding": {
    "transformer": {
      "model.layers.0.mlp.gate_proj.weight": {
        "block": "w.0",
        "encoding": {
          "codec": "int4-sym-g",
          "packing": { "blockElements": 8, "blockBytes": 4, "alignBytes": 4 },
          "rowAxis": 0,
          "groupSize": 32,
          "scale": { "block": "s.0", "dtype": "f32" }
        }
      }
    }
  },
  "assets": {
    "ple.values.0": { "block": "a.0", "role": "ple-values", "length": 33546240 },
    "ple_index": { "block": "a.1", "role": "ple-index", "length": 165 }
  },
  "provenance": {
    "license": "gemma",
    "notice": "NOTICE",
    "upstreamRevision": "…"
  }
}
```

| 欄           | 型       | 必須 | 上限・規則                                                                                                                                 |
| ------------ | -------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `format`     | string   | 必須 | 固定 `"karume-model"`                                                                                                                      |
| `version`    | integer  | 必須 | 固定 `1`                                                                                                                                   |
| `codecs`     | string[] | 必須 | 束縛表が使う codec 登録名の集合と**完全一致**。**重みを 1 バイトも取る前**に台帳と突合する（読み手は各 `encoding.codec` も台帳と照合する） |
| `parts`      | 配列     | 必須 | **添字 1 以上だけ**・1 個以上・**1024 個以下**（§10）。§4.2                                                                                |
| `blocks`     | 配列     | 必須 | 0 個以上・**65,536 個以下**（§10）。§4.1                                                                                                   |
| `binding`    | object   | 必須 | グラフ名 → initializer 名 → 供給（§5）。キーは `graphs` のキーの部分集合                                                                   |
| `assets`     | object   | 必須 | 資産名 → `{ block, role }`。無ければ空オブジェクト                                                                                         |
| `provenance` | object   | 必須 | §2.3                                                                                                                                       |

`parts[]` の要素:

| 欄       | 型     | 必須 | 規則                                                                                           |
| -------- | ------ | ---- | ---------------------------------------------------------------------------------------------- |
| `index`  | u32    | 必須 | **1 始まり**の連番（`index = 配列添字 + 1` MUST）。**part 0 は載せない**                       |
| `length` | u64    | 必須 | part のバイト長。part 1 は実長、part 2 以降は §10 の part 長集合の 1 つ以下                    |
| `sha256` | string | 必須 | part 全体のバイト列の sha256。**公開・再梱包の突合用**であり、実行時の完全性には使わない（§7） |

**`parts` が添字 1 から始まる理由**: part 0 はモデル記述そのものを含むので、part 0 の sha256 を
モデル記述に書くと**自己参照**になる（書いた瞬間にハッシュが変わる）。したがって **part 0 の
完全性は外側の期待 hash + 長さが張る**（§7 の①）。manifest `karume/5` の FileRef は
**part 0 を含む全 part** の `size` / `sha256` を持つので、外側から見た完全性に穴は空かない。

**part 1（const 領域）は const が 1 本も無いときも長さ 0 で必ず宣言する**（§3）。part の添字が
「1 = const 領域」で常に一定になり、読み手が配置を数え直さずに済む。

`blocks[]` の要素:

| 欄       | 型     | 必須 | 規則                                                                                                      |
| -------- | ------ | ---- | --------------------------------------------------------------------------------------------------------- |
| `id`     | string | 必須 | block id（§0）。`const.blocks[].id` と素集合 MUST・重複禁止                                               |
| `part`   | u32    | 必須 | **2 以上**（part 0 は descriptor・part 1 は const 専用）                                                  |
| `offset` | u64    | 必須 | その part の先頭からの offset。64 の倍数 MUST                                                             |
| `length` | u64    | 必須 | 4 の倍数 MUST・§10 の block 上限以下                                                                      |
| `sha256` | string | 必須 | block のバイト列（padding 含む）の sha256                                                                 |
| `role`   | string | 必須 | `"weight"` / `"scale"` / `"zero-point"` / `"asset"` のいずれか（`"const"` は const 目次側にしか現れない） |

`assets` の値:

| 欄       | 型     | 必須 | 規則                                                                                                                                                                                                       |
| -------- | ------ | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `block`  | string | 必須 | `blocks[].id` を指す。その block の `role` は `"asset"` MUST                                                                                                                                               |
| `role`   | string | 必須 | **models 側の解釈者名**（`ple-values` / `ple-scales` / `ple-index` / `rope-base` / `style-vectors` …）。**runtime は解釈しない** — descriptor が持つのは「名前 → block」までである                         |
| `length` | u64    | 必須 | payload のバイト数（論理長）。資産は shape を持たないので自分で宣言する。その block の `length` は**これを 4 の倍数へ切り上げた値 MUST**（詰め物の量まで宣言で閉じ、消費側が末尾の 0x00 を推測で剥がない） |

### 2.3 `provenance`

| 欄                 | 型     | 必須   | 規則                                                              |
| ------------------ | ------ | ------ | ----------------------------------------------------------------- |
| `license`          | string | 必須   | ライセンス**識別子**。本文は載せない（本文はリポ直下 — ADR 0071） |
| `notice`           | string | 省略可 | NOTICE ファイルへの参照（パス断片）。本文は載せない               |
| `upstreamRevision` | string | 省略可 | 上流 checkpoint の revision                                       |
| `writer`           | string | 省略可 | 書き手の識別（`karume` の版など）                                 |

`provenance` は **`krg` には引き継がれない**（§9）。単一 `krm` を 1 ファイルとして持ち出したときに
出所が辿れることが目的で、`krg` は「計算契約」であって配布物の出所ではないためである。

## 3. const 領域

- const 領域は**グラフ記述が所有する**。目次は `const.blocks`・initializer への束縛は
  `const.constants`（ともに §2.1）で、**offset は const 領域の先頭からの相対**である。
- 中身は**重み非依存の定数**: 定数畳み込みの frontier で焼かれた値と、config 由来の値
  （RoPE の `inv_freq` など）。**重み由来の scale は const ではない**（scale はモデル記述側の
  block）。
- 分割形では **part 1 が const 領域そのもの**（part 1 の先頭 = const 領域の先頭）。単一形では
  part 0 の直後（64 B 整列後）から `const.length` バイトが const 領域である。
- const が無いときは `const.length = 0`・`const.blocks = []`・`const.constants = []` とし、
  分割形では **part 1 を長さ 0 で置く**（part 番号の意味を配置に依らせないため）。
- **長さ 0 の part の前に詰め物を挿まない** MUST。挿むと、const が空の `krm` から抽出した `krg`
  と、同じグラフを直接書いた `krg` がバイトでずれる（§9 のバイト同一が壊れる）。

const は小さいとは限らない。実測（ミラー 447 本の集計）: gemma4 e2b i4 で総量の 0.08 %・
birefnet 2048 f32 で 29.3 MB（2.5 %）・sbv2 F1 front i4 で 2.1 MB（**29.7 %**）。したがって
準備時に取るのは part 0 だけで、**const は要るときに取る**（§11）。

**const block 1 本が block 上限（§10）を超えたときは、書き手が fail loudly で止まる。** const は
piece 分割の機構を持たない（piece 列を表せるのは重みの束縛表だけで、`krg` はそれを持たない —
§9）ので、上限を超えた const は**表現できない**。現状の最大は birefnet 2048 の f32 定数 29.3 MB
で、32 MiB まで約 2.7 MB の余裕しかない。

**未決**（実需が出たときの裁定・今は逃げ道を作らない）: ①const にも piece 列を許す
（= `krg` に最小限の重み束縛表が戻る）②const だけ block 上限を別に持つ ③大きい定数を資産
（`assets`）扱いにして遅延取得させる。

## 4. block と part

### 4.1 block

**block は独立した取得・検証・解放の単位**である。

- **block は part をまたがない** MUST。
- **block 長 ≤ 32 MiB = 33,554,432 B**（§10・独立定数）。**「以下」であって「未満」ではない**。
- **block 先頭は 64 B 整列**（∴ offset は自動的に 4 の倍数）、**block 長は 4 の倍数** MUST。
- 末尾 padding は**書き手が焼く**。padding バイトは **0x00 固定** MUST。
  - payload のバイト長は codec と宣言 shape から決まる（§6）。これを `payloadBytes` とすると、
    `payloadBytes ≤ length` かつ `length − payloadBytes < 4` MUST。
  - `sha256` は padding を含めた `length` バイト全体に対して取る。これにより**再梱包の
    ビット同一が padding を含めて定義される**。
- 同一 part 内で block は互いに重ならず、part 長を超えない。block 間の隙間は許す（整列 padding）
  が、その隙間も 0x00 MUST。
- **詰め物を覆うハッシュは 2 段構えである**:

| 詰め物の位置       | 覆うハッシュ                         |
| ------------------ | ------------------------------------ |
| block 末尾         | **その block の `sha256`**（対象内） |
| block 間 / part 間 | **その part の `sha256`** だけ       |

- 読み手は block ごとに独立に digest でき、GPU へ上げたら独立に解放できる。

block 上限を 32 MiB **以下**とする根拠は 3 点である（CPU 試作 ① / ② の実測・2026-09-22）:

1. **割っても digest の総費用は増えない。** 256 MiB を 5 巡の中央値で測ると、一括 digest が
   277.4 ms / **923 MiB/s** に対し、32 MiB × 8 の block 別（`subarray` を渡す）が 272.9 ms /
   **938 MiB/s**（比 0.984）である。「block 単位で検証する」設計に digest 側の言い訳は要らない。
2. **実資産の最大重みがちょうど上限に乗る。** anima transformer の 2048×8192 f16 重みは
   33,554,432 B ちょうどで、**56 本**がこの値に乗る。上限が「以下」なら piece 分割は **0 本**、
   「未満」なら同じ **56 本**が分割される。block 長の分布は 32 MiB×56 / 24 MiB×1 / 8 MiB×169 /
   4 MiB×56 / 3 MiB×84 / 2 MiB×1 / 1 MiB×85 / 272 KiB×1。
3. **const block は piece 分割できない**（§3）ので、birefnet 2048 の f32 定数 **29.3 MB** が
   収まる必要がある。32 MiB まで約 2.7 MB の余裕しかない。

実装規則:

- digest には **`subarray` を渡す**（`slice` コピーを挟むと 943 → 641 MiB/s）。
- 並行 digest は効く（16 MiB × 16 を `Promise.all` で **3,702 MiB/s** = 69.2 ms。逐次 16 MiB の
  2.07 倍・256 MiB 一括の 4.0 倍）。一括 digest ではこれが取れない。
- ブラウザは未測。Chrome は digest の入力を Blink 内部へ全量コピーする
  （`packages/hub/src/fetch.ts` の `BYTE_BUDGET` の記録）ので、block 化はブラウザでこそ効くはずである
  （**推測** — コピー量が block 長で頭打ちになる）。

**将来の見直し（上限を 16 MiB へ下げる）**: サイズを交互に回した実測では 16 MiB = 1,784 MiB/s に
対し 32 MiB = 938 MiB/s で、速さだけを見れば 16 MiB が有利である。下げるときの副作用は 2 つ —
①**const の上限が先に壊れる**（birefnet の 29.3 MB が入らなくなり、§3 の未決 3 案のどれかが要る）
②**block 件数と descriptor が倍に膨らむ**。切り替えは定数 1 つである。段 3 の RAM ピーク実測は
part 長だけを測り、block 上限は測らなかったので、見直しは段 6（新 bit 幅・Range 取得の回）へ
持ち越す。

**撤回した根拠**: 草案は「32 MiB ちょうどまで 1,865 MiB/s・33 MiB から 948 MiB/s と半減する」
という崖を block 上限の根拠にしていたが、再実測で**再現しなかった**。あの数はサイズを昇順に
連続計測したときだけ出る測り方の産物である（31.0 → 31.5 → 31.94 → 31.999 → 32.0 MiB と続けて
測ると 32 MiB が 1,840 MiB/s を出す）。16 MiB と交互に回すと下表のとおりで、**崖は 32 MiB の
手前にある**。

| block 長        | 交互計測の throughput |
| --------------- | --------------------: |
| **16 MiB**      |       **1,784 MiB/s** |
| **32 MiB**      |         **938 MiB/s** |
| 40 MiB          |             959 MiB/s |
| 64 MiB          |             933 MiB/s |
| 256 MiB（一括） |             923 MiB/s |

原因は未特定（**推測**: 32 MiB は glibc の mmap 閾値の上限と同値で、この境界から先は確保の
たびに mmap / munmap が走る）。**サイズを振る digest 計測は交互に回さないと結論が逆になる。**

### 4.2 part

**part は配信粒度**である（ホスト RAM の器ではない — それは block）。

- part 0 = `[ヘッダ][グラフ記述][モデル記述]`、part 1 = const 領域、part 2 以降 = 重み / 資産の
  block。
- **part 長は書き手が選ぶ**: `{256, 512, 768, 1024} MiB` のいずれか（既定 **256 MiB**）。
  part 0 / 1 は実長で、この集合の制約を受けない（ただし §10 の「part 長の天井」には従う）。
  なお **part 0 はモデル記述の `parts` に載らない**（自己参照になるため — §2.2）。
- **区間読みを要する資産は専用 part に単独で置く** MUST（1 block = 1 part）。PLE の `ple-values` /
  `ple-scales` の block 列がこれに当たる（1 token あたり値 8,960 B + scale 140 B = 9,100 B を 2 回に
  分けて読む — ADR 0109 決定 4・ADR 0108 追記 3 の 2）。取得元によっては区間読みが scan
  （offset 比例・Deno の既定経路）なので、同居させると「part 先頭からの走査」になって
  区間読みの利点が消える。
- manifest と descriptor の**両方**が各 part の長さを宣言する。読み手は 1 バイトも取る前に
  宿主 RAM を見積れる（ADR 0089 の流儀 — 取得元の型で閉じ方が違う点は §11）。
- **既定は 256 MiB**（ADR 0108 追記 5 の 1 — 段 3e の実測）。seek 型の取得元（ブラウザの Blob・
  ローカルの区間読み）は block ごとに読むので、ホスト RAM のピークは part 長に依らない。scan 型
  （HF 経由 — 取得層の戦略 stream）は part を 1 度に読んで block に切り、hub の保持枠が part を握るので、ピークが
  おおむね part 長とともに伸びる。既定を上げて得るのは part 本数だけで、区間読みの資産は専用 part
  なので本数もほとんど減らない。取得面は ADR 0109 決定 7。
- part 件数 ≤ 1024（旧 `MAX_SHARDS` の値を継承 — 定数は `MAX_PARTS`: Python 正本
  `tools/exporter/src/karume/container.py`・hub `packages/hub/src/manifest.ts`・runtime
  `packages/runtime/src/format/container/limits.ts`）。

**旧 `SHARD_BYTE_LIMIT` = 256 MiB**（旧 shard 配布形の shard 長の上限）の根拠 3 点の行き先:

| #  | 旧根拠                                                             | 行き先                                                                |
| -- | ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| R1 | ホスト RAM ピーク = 定数 + 最大 shard 1 本                         | **block 上限へ移る**                                                  |
| R2 | ファイル数・リクエスト数が hub の 4 並列と読み手上限 1024 本の内側 | **part に残る**                                                       |
| R3 | Chromium の単一 ArrayBuffer 上限 2,145,386,496 B の十分下          | **block 上限へ移る**（part 全体を 1 本の ArrayBuffer に載せなくなる） |

## 5. 束縛表と規則 ①〜⑤

束縛は **`(グラフ名, initializer 名)` で識別する**。値は**供給形の 1 形**（借用は束縛表に載せない — 下）:

**供給形**

```jsonc
{
  "block": "<block id>",
  "encoding": {/* §6.1 */},
  "pieces": [{ "block": "<block id>", "rows": [0, 4096] }]
}
```

| 欄         | 型     | 必須                    | 規則                                            |
| ---------- | ------ | ----------------------- | ----------------------------------------------- |
| `block`    | string | `pieces` が無いとき必須 | 実体 1 本の block id。`role` は `"weight"` MUST |
| `encoding` | object | 必須                    | §6.1                                            |
| `pieces`   | 配列   | 省略可                  | 2 個以上。**`block` とは排他**                  |

`pieces[]` の要素は `{ block, rows: [begin, end) }`。`rows` は**先頭次元**の半開区間。

**借用（shared）は束縛表に載せない。** 別 Session からの借用（現行 `sharedWeights` —
`packages/runtime/src/runtime/session-types.ts` の `sharedWeights`）は IR 側の `{ "shared": true }` 宣言だけが
持ち（[ir-v2.md](ir-v2.md)「共有 initializer」— 借り手の名前 = 貸し手の initializer 名）、束縛表の
**突合集合の外**にある。束縛表に書くと余剰として fail loudly。

### 規則

- **① 1 block ≤ 1 binding**（1 実体 1 initializer）。同じ block id を 2 つの binding が指すのは
  fail loudly。1 つの initializer を**複数のノードが使う**のは可（これは別の話）。
  scale / zero-point の block も同様に 1 つの binding にしか属さない。
- **② 全 block の長さと offset が 4 の倍数**。offset は 64 B 整列規則（§4.1）が包含する。
  長さについては、旧規則「末尾でない piece の block 長は 4 の倍数」を**一般化**したものである。
  末尾 padding を書き手が焼く（§4.1）ので、読み手側の詰め分岐は要らない。
  - **この規則は重み block だけでなく全 block に掛かる** — const も資産も含む。長さが 4 の
    倍数でないもの（bool 表・奇数長の f16 など）は**詰め物込みで宣言し**、**消費側は宣言 shape
    から論理長を導く**（block 長を論理長として使わない）。
  - **詰め物を掛けてよいのは、丸ごと 1 本の block と piece 列の末尾だけ**である。**中間 piece に
    掛けてはならない**（掛けると次の piece の先頭を潰す — `session-build.ts` の `tailAligned` の注記が
    そう書いている）。したがって**中間 piece のバイト長は元から 4 の倍数 MUST** で、書き手は行の
    刻みを `4/gcd(rowBytes, 4)` に丸めて切り、丸め切れないときは fail loudly。
- **③ companion scale の block は実体と同一 part**。piece 列のときは **piece 1 と同一 part**。
  zero-point の block も同じ規則に従う。
- **④ pieces は行範囲を隙間なく被覆し、part は添字順に非減少**（MUST — 構築側は piece 1 が最初の
  part にある前提で簿記する。逆順は `openContainer` で拒否 — 2026-09-25）。`rows` は昇順・`begin[0] = 0`・
  `end[k] = begin[k+1]`・`end[最後] = shape[0]` MUST。
  **`encoding.rowAxis != 0` の initializer は piece 分割を許さない**（scale の行範囲が piece の
  行範囲に対応しないため — 今日の exporter も `conv_transpose1d` の行分割を禁じている）。
- **⑤ 宣言 shape / dtype と実物が完全一致する**。不足も余剰も**全件列挙で拒否**する。
  - `binding` のキー集合は、そのグラフの `initializers` のうち `shared` でないものの集合と
    **完全一致** MUST。
  - `blocks` / `const.blocks` のうち、どの binding / `assets` からも参照されない block は
    **余剰**として拒否する。
  - payload バイト長は `encoding` と宣言 shape から決まる（§6）。実測と違えば拒否。

## 6. `encoding` と codec 台帳

### 6.1 `encoding` の欄

```jsonc
{
  "codec": "int4-sym-g",
  "packing": { "blockElements": 8, "blockBytes": 4, "alignBytes": 4 },
  "rowAxis": 0,
  "groupSize": 32,
  "scale": { "block": "s.0", "dtype": "f32" },
  "zeroPoint": { "block": "z.0" }
}
```

| 欄                      | 型     | 必須   | 規則・上限                                                                                                 |
| ----------------------- | ------ | ------ | ---------------------------------------------------------------------------------------------------------- |
| `codec`                 | string | 必須   | 台帳の登録名（§6.3）。**台帳に無い名前は重み取得前に拒否**                                                 |
| `packing.blockElements` | u32    | 必須   | 1 以上。1 packing block が運ぶ**要素数**                                                                   |
| `packing.blockBytes`    | u32    | 必須   | 1 以上。1 packing block の**バイト数**                                                                     |
| `packing.alignBytes`    | u32    | 必須   | `1 / 2 / 4 / 8 / 16 / 64` のいずれか。payload 先頭の整列要求                                               |
| `rowAxis`               | u32    | 量子化 | `0` または `1`。**行の軸**。台帳の `scale` が required の codec でのみ書く（非量子化で書くと fail loudly） |
| `groupSize`             | u32    | 量子化 | 1 以上・行長を割り切る MUST。per-channel の codec では行長に**等しい** MUST                                |
| `scale.block`           | string | 量子化 | scale の block id。`role` は `"scale"` MUST                                                                |
| `scale.dtype`           | string | 量子化 | 台帳の `scale.dtype` 受理集合の 1 つ（初版は `"f32"` のみ）                                                |
| `zeroPoint.block`       | string | 省略可 | 台帳の `zeroPoint.allowed` が真の codec でのみ書ける（初版は 4 種とも偽 ⇒ 常に省略）                       |

**bit 数は宣言に置かない**。`packing.blockElements` / `packing.blockBytes` からの派生値である
（`bits = blockBytes · 8 / blockElements`）。`bits` を宣言に置くと非整数 bpw が表せず、payload
バイト長も決まらない。

**payload バイト長の算出**（旧配布形の `numel × bits / 8` 厳密一致の一般化 — 実装は
`packages/runtime/src/format/container/codecs.ts` の `payloadBytes`）:

```text
numel % blockElements == 0                   ← MUST
payloadBytes = numel / blockElements * blockBytes
```

**`packing` は台帳の写しであり、読み手にとっては照合用である。** 読み手は宣言された
`packing` が台帳エントリの値と**一致することを検査し、食い違えば fail loudly** にする
（導出できる事実を独立に更新される欄として持つのではなく、cross-package 不変条件の突合点として
置く — hub の `manifest.ts` が `MAX_PARTS` / `MAX_DESCRIPTOR_BYTES` を Python 正本（`container.py`）
と同値に置き、突合点として名指ししているのと同じ流儀）。これは「別の版の台帳で書かれた資産」を静かに受理しないための門である。

**`rowAxis` を宣言に出す**のがこの版の実質的な改善点である。IR v1 では、i8 の per-channel scale の
チャネル軸は宣言に書かれておらず、**消費側 op から導いていた**
（`packages/runtime/src/runtime/plan.ts` の `weightChannelAxes` — 今は宣言との突合として残り、消費 op が
食い違うと落ちる — §13.2）。`rowAxis` を宣言に出すと、**宣言だけで scale の意味が閉じる**。

**scale の形は rank 2 group 形に一本化する**:

```text
scaleShape = [ shape[rowAxis], rowLength / groupSize ]
rowLength  = numel / shape[rowAxis]
```

**要素数 0 の退化形**（`in_features = 0` など・行長 0）: per-channel の `groupSize` は 1 以上 MUST を
満たせないので **1**、group 数は **1**（per-channel scale は行ごとに 1 本あり、旧配布形の `[rows, 1]` と
一致する）。group codec（`int4-sym-g`）の `rowAxis` は **0** だけ（展開カーネルと `decodeI4` が
先頭次元を行とする）。

旧配布形で i8 だけが使っていた「重みと同 rank の keepdim broadcast 形」は廃止した（読むのは移行 CLI
だけ — §12）。i8 の per-channel は
「行 = `rowAxis` の次元・group 長 = 行長」とみなすと `[shape[rowAxis], 1]` になり、i2 が既に
採っている形（`[N,1]`）と同じものになるからである。

### 6.2 codec 台帳のエントリ

台帳は**リポ内の不変なデータ**である。import 時登録はしない（横断の不変条件「全モジュール
副作用ゼロ」に反する）。

```text
{
  name,                                                  // 宣言に書く登録名
  packing:   { blockElements, blockBytes, alignBytes },  // 宣言バイト長と整列の正本
  levels:    { kind, offset, min, max, unused? },        // 値の集合
  scale:     { required, dtype, form: "rank2-group" },
  zeroPoint: { allowed },
  decodeCpu,                                             // 唯一の CPU 展開
  executableOps,                                         // 圧縮のまま常駐できる op の集合
  wgsl?,                                                 // 実行経路（無ければ宣言のみ受理）
}
```

**3 つの軸を分けて報告する**:

| 軸                                 | 判定するもの                     | 判定材料                                    |
| ---------------------------------- | -------------------------------- | ------------------------------------------- |
| ① 宣言として読めるか               | 受理 / 拒否                      | `packing` + `scale`（台帳に名前があれば真） |
| ② 実行できるか                     | 実行 / capability 不足で列挙拒否 | `wgsl` の有無                               |
| ③ どの op で圧縮のまま常駐できるか | packed 常駐 / ロード時 CPU 展開  | `executableOps`                             |

この 3 分離は**現行が既にそうなっている**: `bf16` は①のみ（宣言は valid だが
`RUNTIME_SUPPORT.storage` に無い — `packages/runtime/src/ops/contracts.ts`）、`i2` は①②③だが
③は linear / embedding だけ。**新しい codec を「宣言だけ先に受理する」道が既に通っている**。

`executableOps` は、今日 3 本に分かれている純関数
（`plan.ts` の `eligibleCompressedInitializers` / `i2EligibleInitializers` /
`i4EligibleInitializers` — 後の 2 本はほぼ同文）を 1 欄へ畳む。

`decodeCpu` と `wgsl.unpackScalar` が**同じエントリに並ぶ**ことで、「pack 順の正本は 1 箇所」
という現行の MUST（`packages/runtime/src/format/i4.ts` 冒頭の MUST・
`tools/exporter/src/karume/emit.py` の `pack_int4`）が構造として守られる。

### 6.3 初版の codec 台帳（4 種）

値はすべて現行実装からの**逐語移送**である（新しい値を 1 つも作らない）。

**登録名（`int8-sym` / `int4-sym-g` / `int2-off` / `ternary`）は確定した綴りである（2026-09-22
裁定）。** 本書が置いた名前で、上流に由来する名前ではない。台帳の登録名は**資産に焼かれる**（後から
変えると移行がもう 1 回要る）ので、以後は変えない。

| 項目                    | `int8-sym`                                              | `int4-sym-g`                                                                      | `int2-off`                                                    | `ternary`                            |
| ----------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------ |
| 旧 `storage.dtype`      | `i8`                                                    | `i4`                                                                              | `i2`                                                          | （新設）                             |
| 量子化値 `q` の集合     | `[−127, +127]`（**−128 は使わない**・255 準位）         | `[−7, +7]`（**−8 は使わない**・15 準位）                                          | `[−2, +1]`（4 準位すべて）                                    | `{−1, 0, +1}`（3 準位）              |
| 格納コード `u`          | `u = q`（符号付き 8bit）                                | `u = q + 8`（`[1,15]`・0 未使用）                                                 | `u = q + 2`（`[0,3]`）                                        | `u = q + 2`（`[1,3]`・**0 未使用**） |
| `levels.kind`           | `int-sym`                                               | `int-sym`                                                                         | `int-offset`                                                  | `int-sym`                            |
| `levels.offset`         | 0                                                       | 8                                                                                 | 2                                                             | 2                                    |
| `packing.blockElements` | 1                                                       | 8                                                                                 | 16                                                            | 16                                   |
| `packing.blockBytes`    | 1                                                       | 4                                                                                 | 4                                                             | 4                                    |
| `packing.alignBytes`    | 4                                                       | 4                                                                                 | 4                                                             | 4                                    |
| 派生 bit / 要素         | 8                                                       | 4                                                                                 | 2                                                             | 2                                    |
| 詰め順（1 u32 語）      | 4 要素・`unpack4xI8(w[i>>2])[i&3]`                      | 8 要素・バイト内で要素 `2i` = **下位** nibble / `2i+1` = **上位**                 | 16 要素・**下位 2bit から**順に `(w[i>>4] >> ((i&15)*2)) & 3` | **`int2-off` と完全に同一**          |
| `scale.dtype`           | `f32`                                                   | `f32`                                                                             | `f32`                                                         | `f32`                                |
| `groupSize`             | 行長（per-channel）                                     | **2 冪かつ ≥ 16**（既定 32）                                                      | 行長（per-channel）                                           | 行長（per-channel）                  |
| `rowAxis`               | 0（**`conv_transpose1d` だけ 1**）                      | 0                                                                                 | 0                                                             | 0                                    |
| `zeroPoint.allowed`     | 偽                                                      | 偽（`offset 8` を「zero_point 省略時の既定」と読む予約は ADR 0069 決定 3 が保持） | 偽                                                            | 偽                                   |
| 宣言 shape の追加条件   | 無し                                                    | 行長 `% groupSize == 0`・`numel` は偶数                                           | 正の rank 2 `[N,K]`・**`K % 16 == 0`**                        | `int2-off` と同一                    |
| **復元式（CPU）**       | `fround(q · s)`                                         | `fround((u − 8) · s)`                                                             | `fround((u − 2) · s)`                                         | `fround((u − 2) · s)`                |
| `executableOps`         | linear / embedding / conv1d / conv2d / conv_transpose1d | linear / embedding / conv1d（`groups == 1`）                                      | linear / embedding                                            | `int2-off` と同一                    |
| `wgsl`                  | 有り                                                    | 有り                                                                              | 有り（`linearCompute` は `f32` のみ）                         | **`int2-off` のものを共有**          |

正本の所在:

- `int8-sym`: 値域 `tools/exporter/src/karume/quantize.py` の `INT8_MAX`（`= 127`・「−128 は
  使わない」）、scale = `clamp(amax / 127, f32 tiny)` は `quantize.py` の `channel_scale`、
  `q = round(w/scale).clamp(−127, 127)` は `quantize.py` の `quantize_to_int8`、CPU 展開
  `packages/runtime/src/format/i8.ts` の `decodeI8`。
- `int4-sym-g`: 値域 `quantize.py` の `INT4_MAX`（`= 7`）、既定 group 長 32 は
  `quantize.py` の `DEFAULT_GROUP_SIZE`、scale = `clamp(amax / 7, f32 tiny)` は
  `quantize.py` の `group_scale`、`q = round(w/scale).clamp(−7, 7)` は `quantize.py` の
  `quantize_to_int4`、pack 順の正本 `tools/exporter/src/karume/emit.py` の `pack_int4`、offset 定数
  `emit.py` の `INT4_OFFSET`（`= 8`）、group scale 形 `packages/runtime/src/format/container/codecs.ts` の
  `groupScaleShape`（合流層・常駐プランナ・構築・CPU 展開が共有する 1 本）、CPU 展開
  `packages/runtime/src/format/i4.ts` の `decodeI4`。
- `int2-off`: 形の条件 `packages/runtime/src/format/i2.ts` の `isI2Shape`、CPU 展開 `i2.ts` の
  `decodeI2`、pack `emit.py` の `pack_int2`（値域 `[-2,1]` 外を拒否）、scale は F32 の `[N,1]`・group 不可（ADR 0097
  追記 1）。
- 宣言規則は 2 箇所に分かれる。scale / `groupSize` の有無は descriptor の読み手
  `packages/runtime/src/format/container/descriptor.ts`（`parseEncoding`）と、入力形が違うメモリ内容器
  `packages/runtime/src/format/container/memory.ts` が見る。`groupSize` の値域・行長の整除・i2 系の宣言
  shape・scale 長は合流層 `packages/runtime/src/format/container/bind.ts`（`bindDeclarations`）が見る。
  Python の鏡像は `tools/exporter/src/karume/verify.py` の `bind_graphs` で、その `_plan_supply` は有無も
  見る。

**`ternary` が別名の codec である理由**（値は `int2-off` と 1 つも違わないのに分ける理由）:

- **runtime の追加は 0 行**。`{−1, 0, +1}` は `int2-off` の値域 `[−2, +1]` の**部分集合**なので、
  詰め方も復元式も WGSL も CPU 展開も `int2-off` のものをそのまま使う。追加するのは exporter 側の
  **absmean 量子化器 1 本**だけである（**未実装** — ADR 0108 の段 6。現状の exporter は `ternary` の資産を作れない）。
- それでも分けるのは、**資産を識別できるようにする**ため。逆向き（i2 資産を三値として読む）は
  成立しない: gemma4-qat e2b の I2 テンソル先頭 4 MiB を 2bit コードで数えた実測で、
  `q = −2` が **5.8〜7.4 %** 出ている（`model.lm_head.weight` 7.447 % /
  `layers.15.mlp.down_proj.weight` 5.766 % / `layers.15.mlp.gate_proj.weight` 7.230 %）。
  黙って読み替えると全要素の 6 % 前後が別の値になる。
- `ternary` 宣言の追加条件: payload の全コードが `{1, 2, 3}` に入ること（コード 0 の出現は
  fail loudly）。これが「三値である」という主張の検査可能な中身である。

### 6.4 奇数 bit の予約 — ビット面（bit-plane）分解

3 / 5 / 6 / 7 bit は**台帳のエントリ形だけを予約する**。実装は需要が出た幅から行う。

方式: **32 要素を `bits` 本の u32 へ「平面」で置く**（語 `b` のビット `j` = 要素 `j` の
第 `b` ビット）。

- 格納は**厳密に `bits` / 要素**（捨てビットは 0 で、奇数 bit でも膨らまない）。
- 平坦添字は常に 2 冪（32 要素 / 面）で割れるので、「平坦添字から語内位置をシフトで割る」
  不変条件と「行頭が語境界に来る」不変条件が両方保たれる。
- `packing` は `{ blockElements: 32, blockBytes: 4 * bits, alignBytes: 4 }` と書ける。
- unpack 費用（**推測** — 未実測）: 要素あたり `bits` 回の `(w_b >> j) & 1` と `bits − 1` 回の
  OR / shift ≈ **3·bits 命令**（3 bit で 9 命令・現行 i4 の 4〜5 命令の約 2 倍）。

**採らない詰め方**（理由は ADR 0108 の却下案 4 / 5）:

- 「u32 語に n 個詰めて余りを捨てる」— 実効 bit が膨らみ（3 / 5 / 6 bit で +6.7 %・7 bit で
  +14.3 %）、1 語あたりの要素数が 2 冪でなくなるので group scale の添字計算がシフトから
  乗除算に変わる。
- base-243（GGUF TQ1_0 の 5 trit / byte）— 2 bit 詰めに対し **−15.6 % しか縮まない**一方で、
  5 要素粒度が 2 冪でないため不変条件が 2 つ壊れる。行長に `% 5` の整除条件が要り、gemma4 の
  実形（1536 / 2048 / 6144 / 12288）は**どれも 5 で割れない**。

**低 bit 化は速度の理由にしない。** decode は本機で帯域律速になっておらず（帯域利用率
11.5 / 25.5 % の実測）、効くのは**メモリ**である（gemma4 e2b の i4 g32 → 三値 2 bit で
格納 −40 %・f16 scale 化だけで −10 %・group 32 → 128 で −15 %）。低 bit codec は packed int8
活性（ADR 0105）が前提である。

## 7. 完全性 — ハッシュの役割 3 分離

**descriptor は自分の正しさを証明できない。** したがって**外側の期待 hash + 長さで先に検証する**:

1. 読み手は外側（manifest `karume/5` の `container.descriptor`、または `openContainer` の呼び手が第 2 引数
   `expect`〈`DescriptorExpectation`〉で渡す pin）から **2 文書それぞれの期待 sha256 と期待バイト長**を受け取る（ADR 0109 決定 3 — part 0
   ファイルの sha256 とは別の事実）。
2. part 0（または単一形の先頭）を取得し、ヘッダを読んで 2 文書の長さを得る。
3. **2 文書のバイト列を期待値と突合してから JSON を parse する**。突合前に parse しない。
4. parse 後に §2 / §5 の宣言検査を全部通す（**重みを 1 バイトも取る前**）。

実行時の完全性は **descriptor と block の sha256** で張る。ただし **block の digest を掛けるのは未検証の
取得元だけ**である（ADR 0108 追記 2 の 3 / ADR 0109 決定 7）:

| 経路                                                                       | digest の回数                                                                          |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| cold（HF 経由の初回取得）                                                  | 取得層が part 全量を流しながら **1 回**（記録ハッシュを焼く）。block の digest は 0 回 |
| warm（キャッシュヒット）                                                   | **0 回** — 記録ハッシュの文字列比較だけ                                                |
| 未検証の取得元（`openContainer({ kind: "bytes" })`・ローカルディレクトリ） | **block ごとに一括 digest**（32 MiB 以下なので §4.1 の速い側に収まる）                 |

`BlockSource` が「検証済み」を名乗り、`readBlock` は名乗らない取得元にだけ digest を掛ける。
warm で digest を走らせないのは現行の規律の継承である（`packages/hub/src/fetch.ts` の `BYTE_BUDGET` の NOTE —
キャッシュヒットで GB 級の digest を起こさない）。**「0 回」は重みの block と資産の payload について**
であり、上の手順 3（2 文書を期待値と突合してから parse する）は cold でも warm でも開くたびに掛かる
（descriptor は 32 MiB 以下・実資産では数百 KB）。取得層の cold の part 全量の逐次 sha256 は純 TS 実装
なので、`crypto.subtle.digest` の計数（`tools/ram-peak`）には現れない。block の sha256 は段 6 の Range 取得で「届いた分だけ
検証する」ための契約として残る。

**ファイル全体の sha256**（`parts[].sha256`）は**公開・再梱包の突合用**として分離する。
実行時には使わない。

**ハッシュの役割 3 分離**（同じ欄で兼ねてはならない）:

| 役割                       | 何を指すか                               | いつ決まるか           |
| -------------------------- | ---------------------------------------- | ---------------------- |
| ① **取得物の期待ハッシュ** | 「これから取るバイト列はこれであるべき」 | 取得**前**・外側が持つ |
| ② **派生キャッシュキー**   | 「入力 digest + recipe 版 + 変換設定」   | 生成**前**に決まる     |
| ③ **派生物の内容ハッシュ** | 「生成した結果のバイト列はこれだった」   | 生成**後**に計算する   |

この 3 つを 1 欄で兼ねると、上流重みの取り込み（まだ存在しないものの内容ハッシュが要る形）や
ロード時量子化の派生物キャッシュが表せなくなる。

## 8. 単一形と分割形

**バイト列として同じ並び**であり、「切れ目があるか否か」だけが違う。

|                  | 単一形                                                             | 分割形                                          |
| ---------------- | ------------------------------------------------------------------ | ----------------------------------------------- |
| part 0           | ファイル先頭から。ヘッダ + 2 文書ちょうど（詰め物を含まない）      | 独立したファイル                                |
| part 1（const）  | part 0 の直後（長さ 0 でなければ 64 B 整列まで 0x00 を詰めてから） | 独立したファイル                                |
| part 2 以降      | part 1 の直後・**64 B 整列**で連結                                 | それぞれ独立したファイル                        |
| `parts[].length` | 論理 part の長さ（連結後の区間長）                                 | ファイル長                                      |
| 取得の FileRef   | 1 本                                                               | descriptor 1 本 + parts N 本（manifest が持つ） |

- **`parts` は添字 1 以上だけ**を載せる（part 0 の sha256 を自分に書くと自己参照 — §2.2）。
  part 0 の完全性は**外側の期待 hash + 長さ**が張る（§7）。manifest `karume/5` の `container.parts` は
  **part 0 を含む全 part** の FileRef（`size` / `sha256`）を添字順に持つ（ADR 0109 決定 3）。
- 分割形では長さ 0 の part（const が空の part 1）も **0 バイトのファイルとして書く**。manifest は
  その FileRef を `size: 0` で持ち、hub は取得しない（ADR 0109 決定 3）。
- **長さ 0 の part の前に詰め物を挿まない** MUST。part 絶対 offset は「宣言された part 長 +
  64 B 整列規則」から導くので、長さ 0 の part に詰め物を与えると const が空の `krm` からの
  抽出結果と直接書いた `krg` がずれる（§9）。
- **HF の公式配布は分割形のみ**。
- **ファイル名**（exporter / 移行 CLI の規約 — manifest はこれを FileRef で指す）: 単一形は
  `<stem>.krm`、分割形は `<stem>-NNNNN-of-NNNNN.krm`（part 0 から・5 桁ゼロ詰め・旧 shard と同じ
  綴り規約）、`krg` は `<stem>.krg`。
- 単一形を HF に置くこと自体は「ダウンロード用資産」として禁止しない。**制限は場所ではなく
  取得能力で説明する**（§10 の `openContainer({ kind: "bytes" })` の上限）。
- 単一形 `krm` は `krg` を**内包する**（§9 でそのまま抜ける）。
- 分割形の `krm` は、グラフを内包してもよいし、descriptor に **`graph` の内容参照**
  （`{ sha256, size }`）だけを持って外部の共有 `krg` を指してもよい。参照の取得先は
  **manifest が FileRef で与える**（descriptor は repo の概念を持たない）。`karume/5` にはこの席を
  置かない — 要る段（段 5）で足す（ADR 0109 決定 5）。

## 9. `krg` のバイトコピー抽出

```text
krg = [ヘッダ'][グラフ記述（krm からのバイトコピー）][詰め物][const 領域（krm からのバイトコピー）]
```

- **offset の書き換えは 1 バイトも要らない**。const block の offset が「const 領域の先頭からの
  相対」だからである（§3）。
- 書き換わるのは**ヘッダだけ**: magic が `KRGC` に、model descriptor length が `0` に、
  graph descriptor length はそのまま。詰め物は写さず、`krg` 自身の「ヘッダ + グラフ記述」の長さから
  §8 の配置規則で導き直す（値は 0x00 なので写す中身は無い・const が空なら詰め物も無い）。
- したがって「`krm` から抜いた `krg`」と「最初から `krg` として書いた同じグラフ」は
  **バイト同一**になり、`krg` の同一性を**内容ハッシュ**で判定できる。
- `krg` が**持たない**もの: **重みの束縛表**（重みの要求は `graphs[].values` の宣言 shape /
  dtype から導出する）・`provenance`・重み block・scale block。したがって **`krg` 単独では Session を
  組めない**（`createSessionFromContainer` は「重みの供給が無い」で fail loudly。const 供給と shared
  だけのグラフは例外的に組める）。
- `krg` が**持つ**もの: `const.constants`（const block → initializer の束縛表）と
  `const.length`。どちらもグラフ記述の中にあり、**`krg` 単独で `const.*` を供給できる唯一の
  手段**である（`const` の `encoding` をどこからも読めなくなるのを防ぐ）。
- `krg` が持ち出せる資産は**重み非依存**のものだけ（`rope_base` のような const 領域の付属表）。
- `krg` の定義は「**重み block を持たない**」であって「重み非依存」ではない。QAT のグラフは
  `static_quantize` の attr に活性 scale を焼き込む（ADR 0097 追記 2）ので、`krg` でも重みに
  由来する数を抱えることがある。

## 10. 上限

**それぞれ独立した定数として持つ**（一方を他方から派生させない）。

| 対象                                          |                                   上限 | 根拠                                                                                                                                                                                      |
| --------------------------------------------- | -------------------------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| グラフ記述の長さ                              |                                 32 MiB | 実測の IR は最大 610,143 B / 本・重複除去 47 本で 9.0 MiB。part 0 を取る費用を block 上限と同じ桁に収める（**暫定値**）                                                                   |
| モデル記述の長さ                              |                                 32 MiB | 同上（**暫定値**）                                                                                                                                                                        |
| JSON の入れ子深さ                             |                                     64 | parse の再帰深さを宣言で閉じる                                                                                                                                                            |
| **block 長**                                  |         **33,554,432 B（32 MiB）以下** | §4.1 の根拠 3 点（割る費用ゼロ / 最大重み 56 本がちょうどこの値 / const が piece 分割できない）                                                                                           |
| block 件数（1 コンテナ）                      |                                 65,536 | const 目次とモデル目次の合計（**暫定値**）                                                                                                                                                |
| part 長（part 2 以降）                        | `{256, 512, 768, 1024} MiB` のいずれか | §4.2                                                                                                                                                                                      |
| part 長の天井（part 0 / 1 を含む全 part）     |            1,073,741,824 B（1024 MiB） | part 2 以降の集合の最大値と同じ値を part 0 / 1 にも掛ける — 器の寸法を宣言から見積る式を 1 本に保つ                                                                                       |
| part 件数                                     |                                   1024 | 旧 `MAX_SHARDS` の値を継承（`MAX_PARTS` — §4.2）                                                                                                                                          |
| 1 FileRef のバイト数                          |                                 16 GiB | 現行 `MAX_FILE_BYTES`（`packages/hub/src/manifest.ts`）                                                                                                                                   |
| `graphs` の個数                               |                                     64 | 実資産の最大は sbv2 の 4 グラフ級（**暫定値**）                                                                                                                                           |
| **`openContainer({ kind: "bytes" })` の全量** |                    **2,145,386,496 B** | `MAX_SINGLE_CONTAINER_BYTES`（`packages/runtime/src/format/container/limits.ts`）。Chromium の単一 ArrayBuffer 上限（実測の記録は `packages/hub/src/fetch.ts`）。**この口にだけ残る制限** |

**暫定値について**: `graphs` の個数 64・block 件数 65,536・descriptor の長さ 32 MiB は、実測
（IR 最大 610,143 B / 本・重複除去 47 本で 9.0 MiB・sbv2 で 4 グラフ級）から**余裕を取って置いた
暫定値**であり、実資産からの拘束ではない。**実装時に詰める。**

**上限を派生させない理由**: 旧仕様は block の概念を持たず、RAM ピークも ArrayBuffer 天井も
配信粒度もすべて `SHARD_BYTE_LIMIT` 1 本に載せていた（ADR 0090 決定 2）。1 本に載せると、
どれか 1 つの根拠が動いたときに他の 2 つが巻き添えで動く。

**block 上限を超える const の逃げ道は未決**である（§3 — ①const にも piece 列を許す ②const
だけ上限を別に持つ ③大きい定数を資産扱いにして遅延取得させる）。実需が出るまでは**書き手が
fail loudly** で止まる。

## 11. メモリ契約

**ブラウザの RAM ピーク = 定数 + 取得単位（block）の重ね合わせ**に抑える。

- 重ね合わせは**本数と合計バイトの両方**で制限する（例: 「転送中 1 本 + 受信・検証中 1 本」）。
  本数だけで律速を決めると、ピークが「同時本数 × その時点で一番大きいファイル」で決まって
  しまう（実測: anima turbo i4 の先頭 4 本で計 2.503 GiB を同時前確保 —
  `packages/hub/src/fetch.ts` の `BYTE_BUDGET`）。
- **展開（decode）は別の処理単位にする**。1 bit → f32 は 32 倍・i2 → f32 は 16 倍に膨らむので、
  取得の重ね合わせと展開の重ね合わせを同じ予算で数えない。
- 準備時に取るのは **part 0 だけ**。const（part 1）は要るときに取る。
- **取得単位は part**（ADR 0109 決定 7 — HTTP Range は段 6）: cold は取得層が part 全量を流して検証し
  キャッシュへ落とす（ヒープに part は載らない）。区間読みは seek 型の取得元（ブラウザの Blob・
  ローカルの区間読み）で block ごと、scan 型（HF 経由 — 取得層の戦略 stream）で part を 1 度全量読んで block に切る。
  scan 型の切り出しは hub の保持枠が握る part の器の view で、写しは乗らない
  （`packages/hub/src/container.ts`）。
- **Session 構築は item を 1 本読んでは上げて手放す**（`packages/runtime/src/runtime/session-build.ts`
  の batch ループ）。item は block 1 本で、丸ごとの initializer と piece 1 は同乗する scale の block も
  連れる。`queue.writeBuffer` は呼んだ時点でバイト列を写すので、item への参照はその反復で尽きる
  （フェンスまで握らない — 下の「GPU 転送」）。

**数え方**: Session 構築中の重み由来のホスト RAM を、次の 3 項の和で数える。staging（`writeBuffer` の
溜め込み）は別の軸で、下の「GPU 転送」が律する。

| 項               | seek 型                                                                                                                                                                                                     | scan 型                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| ① 呼び手の持ち物 | `krm` では 0。メモリ内容器（`openMemoryContainer`）では、呼び手が丸ごと供給で渡した `bytes` と companion scale（容器は複製せずに抱える）。`pieces` の `read()` が取得元を通るなら、③ はその取得元の型に従う | seek 型と同じ                                                                                       |
| ② 処理中の item  | block 1 本 + 同乗 scale + 展開席ならその f32 展開結果（格納のビット幅に反比例 — i4 で block の 8 倍・i2 で 16 倍）+ 展開席の piece 列なら持ち越し scale の写し                                              | 展開席の f32 展開結果 + 持ち越し scale の写し（block と同乗 scale は ③ の器の view なので足さない） |
| ③ 取得の保持     | 0（② の block が区間ぶんの tight view そのもの）                                                                                                                                                            | hub の保持枠 1 本（part 長）+ GC を待つ前の器                                                       |
| 宣言から閉じるか | 閉じる                                                                                                                                                                                                      | 閉じない（③ の GC を待つ器の本数）                                                                  |

- **持ち越し scale の写し**: 展開席（CPU で f32 へ展開して載せる席）の piece 列では、companion scale の
  実体が piece 1 の part にしか無い（§5 の規則③）ので、piece 1 で scale 全量を値として写し、列の最後の
  piece まで持ち越す（`session-build.ts` の `carriedScales`）。view のまま抱えると、scale の器
  （seek 型なら scale の block・scan 型なら part の保持枠）が列の最後まで生き残るためである。その代わり
  **piece 列の間だけ scale 全量が 1 本余分に生きる**。piece 1 の時点では同乗 scale の view と写しが
  同時に生きる（scale が 2 本）。`krm` では scale も block 上限（32 MiB）の内側にある。圧縮席（packed の
  まま常駐する席）は写しを取らない — scale は piece 1 でそのまま GPU へ書く。
- 見積りが**宣言だけで閉じる**のは seek 型である: `parts[].length` と `blocks[].length` から、1 バイトも
  取る前に「最大同時ホスト RAM」を計算できる（ADR 0089 の流儀）。scan 型は最大 part に GC を待つ器が
  重なり、器の本数は宣言からは閉じない。宣言から出せるのは下限（保持枠 1 本 = 最大 part 長）までで、
  段 3e の実測では external 最大が最大 part の約 1.5〜4.4 本だった
  （[研究記録](research/2026-09-24-part-length-ram-peak.md)）。
- 「part」はここで 2 つの意味を持つ（hub の part とメモリ内容器の part）。どちらもフェンスの単位で、
  staging の上限を決める。配信粒度として効くのは hub の part だけで、RAM に効くのは scan 型の ③ である。
  ホスト RAM のピークは part の割り方に依らない（staging の置き場は実装依存で、実測は wgpu / Deno だけ）。

### GPU 転送

- **`mappedAtCreation` は使わない**。マップしたバッファは全 block が揃って unmap するまで解放
  できず、「block ごとに取得・検証・解放する」という目的と逆を向く。
  `queue.writeBuffer(buffer, dstOffset, …)` を block ごとに呼ぶ。
- wgpu（Deno）の `writeBuffer` staging は **submit 完了まで解放されない**（実測: `createBuffer`
  2 GiB 後 2,311 MiB → `writeBuffer` 後 4,359 MiB → `queue.submit([])` + 完了待ちで 2,306 MiB —
  `docs/research/2026-08-08-vram-oom-misreport.md:70-76`）。したがって「空 submit +
  `onSubmittedWorkDone`」のフェンス（ADR 0070 決定 3）を **part ごとに 1 回**立てる。フェンスが律する
  のは staging だけで、CPU 側のバイト列は `writeBuffer` が戻った時点で手放してよい（WebGPU 仕様 —
  呼び出しの時点で写す。ADR 0108 決定 9 の追記・ADR 0070 決定 3 の追記。実機の固定は
  `packages/runtime/tests/gpu_write_buffer_copy_test.ts`）。
- **errorScope の粒度は block・フェンス（空 submit + `onSubmittedWorkDone`）の粒度は part**
  （ADR 0108 決定 9 の実測: push/pop は 1.81 µs / 回、フェンスは 13.0 ms / 回。pop だけ block
  ごとにしてフェンスを 256 MiB ごと 1 回にすると、現行相当より速い 0.77 倍）。
- `writeBuffer` で書くバッファは**アリーナのプール対象外**である（`writeBuffer` はキュー順で
  未 submit の先行エンコードを追い越すため — `packages/runtime/src/gpu/arena.ts` 冒頭の MUST と `allocHostWritten`）。
  この性質は block 化しても変わらない。
- **flush-before-destroy** は従来どおり。

## 12. 移行 CLI の契約

旧形式を読む処理は**この CLI にだけ**置く。新パッケージの読み手は `karume/5` と新コンテナ
だけを読み、**両読みは実装しない**。

- **入力**: 旧単一形 safetensors / 旧 shard 列 / 旧 manifest（`karume/4`）。
- **出力**: 新 `krm`（部品単位モードで `--graph` を渡したときは `krg` も）と、リポ丸ごとモードでは
  新 manifest（`karume/5`）。
- **旧入力は保持する**（CLI は入力を消さない・書き換えない）。
- 実装は Python（`tools/exporter/src/karume/migrate.py`）。旧形の読み取りは移行 CLI（`migrate.py` と
  `legacy.py`）にだけ置く。`legacy.py` は safetensors の shard 列と piece キーを、`migrate.py` は旧
  `karume/4` manifest と PLE sidecar / 索引を読む。書く → 読み直して検証 → 据え替えの 3 段は、export の一本道と**同じ 1 本**
  （`karume.publish.publish_container`）を通る。移行 CLI が足すのは旧形の読み取りと、旧形にしか無い
  前提（scale の形・テンソルの過不足・出力先が空であること）の検査だけである。
- **両モード共通の指定**: `--out <dir>`、`--license <識別子>`（必須 — 既定値で出所を偽らない）、
  `--notice <参照>` / `--upstream-revision <revision>`（`provenance` の省略可の欄 — §2.3）、
  `--writer <識別>`、`--part-bytes {256,512,768,1024}`（part 長 MiB・既定 256 — §4.2。集合外は拒否）。
- **`provenance.writer` は既定で書かない**（`--writer` を明示したときだけ容器に載る）。ツールの版を
  持つのは `karume.json` の `generator` 欄 1 箇所である。容器に版を焼くと、移行した容器と recipe が
  直接書いた容器のモデル記述が、版の分だけ永久に食い違う。
- **部品単位モード**（`karume migrate <代表 path | 旧単一形>… --out <dir> --license <識別子>
  [--graph-name <名前>] [--single] [--graph]`）: コンポーネント（グラフ 1 本）単位で、manifest は
  読まないし書かない。代表 path の代わりに手元の現物（`…-00001-of-00002.safetensors`）を渡してもよい。
  グラフ名の既定は親ディレクトリ名で、`--graph-name` で明示できる（位置引数が 1 本のときだけ）。
  規則は §2.1 のとおり weights のキー MUST なので、親ディレクトリ名が部品名と違う置き場では
  `--graph-name` で部品名を名乗る。`--single` は単一形の `krm` を書く（既定は分割形）。`--graph` は
  `krg` も書く。shard 列は代表 path から組むので、ディレクトリを跨ぐ shard 列（sbv2 の `shared/front` /
  `shared/voice` は shard 2 が話者ディレクトリに居る）は組み立てられない — リポ丸ごとモードを使う。
- **リポ丸ごとモード**（`karume migrate --manifest <karume.json> --out <dir> --license <識別子>
  [--cross-repo <owner/name>=<変換済みディレクトリ>@<40 桁 revision>]…`）: 旧 `karume/4` を読み、
  全 (モデル, 部品, dtype) を `krm` へ変換し、`assets` を写して `karume/5` の `karume.json` を書く
  （ADR 0109 決定 8）。shard 列を旧 manifest の宣言から組むので、ディレクトリを跨ぐ列もここで解ける。
  グラフ名は `weights` のキーそのもの（§2.1）。越境参照の列は変換せず、`--cross-repo` が指す変換済み
  ディレクトリの `karume/5` から `container` を引き写す。
- **モード間で併用できない指定**（どれも fail loudly）: `--manifest` と位置引数。リポ丸ごとモードでの
  `--graph-name` / `--graph`（グラフ名はキーが決め、`karume/5` は共有 `krg` の席を持たない — ADR 0109
  決定 5）と `--single`（`container.parts` は part 0 + part 1 の 2 要素以上 MUST・HF の公式配布は分割形
  だけ — ADR 0109 決定 3・§8）。部品単位モードでの `--cross-repo`。
- 自己検査（不変条件 5）は **payload 部**で突き合わせる（block 全体の sha256 は詰め物を含むので
  新旧で一致しない）。出力は `.partial` へ書き、検査を通ってから据え替える（落ちた回は何も残さない）。
- 旧 scale の**形**（keepdim / group 形）が `rowAxis` / `groupSize` から決まる形と一致することを焼く前に
  見る（正方の重みでは per-column の `[1,N]` と per-channel の `[N,1]` がバイト数で区別できない）。

不変条件:

1. **生バイト同一**: 各 initializer の実体 / scale の **payload バイト列**（`numel × bits / 8`
   バイトぶん）が 1 バイトも変わらない。
   - **末尾 padding は新たに焼かれる**（§4.1）ので、「生バイト同一」は payload について言う。
     block 全体（padding 込み）の sha256 は新旧で一致しない。
2. **IR は v1 → v2 へ再 serialize する**（改名と正準直列化が入るので、IR の逐語同一は保証しない）。
3. **codec は既存 3 種へ写像する**: `i8 → int8-sym` / `i4 → int4-sym-g` / `i2 → int2-off`。
   **`ternary` へは変換しない**（値域が部分集合でも、三値であるという主張は量子化器の側が
   するものである）。
4. **決定的**: 同じ入力から**バイト同一の出力**が出る（part への詰め方・block の切り方・JSON の
   キー順がすべて入力から決まる）。
5. **自己検査**: 書いた容器を読み直し、block の **payload 部**の sha256 と、旧 shard から取り出した
   実体の sha256 が **initializer ごとに一致する**ことを CLI 自身が検査してから据える（書けたのに
   読めないものを作らない — `verify.assert_reader_layout` と同じ流儀）。
6. **piece キーの綴りは消える**。旧 `<名前>#00002-of-00003` は `binding[].pieces[].rows` になる。
   名前空間に物理配置が漏れなくなる。旧綴りを解釈するのは移行 CLI の旧形の読み取り（`legacy.py`）
   だけである。

## 13. IR v2 の差分

IR v1 からの差分は **4 点**である（正本は [ir-v2.md](ir-v2.md) — ここは束縛表との対応だけ）。

### 13.1 `initializers[].storage` を束縛表へ外出しし、initializer 名を実体の鍵にする

```jsonc
// IR v1
"initializers": {
  "p_enc_w": { "tensor": "enc.w", "storage": { "dtype": "i4", "scale": "enc.w_scale", "group_size": 32 } }
}

// IR v2 — 名前が v1 のテンソルキー（FQN / const.<hash>）そのものになる
"initializers": {
  "enc.w": {}
}
```

| 旧                                                    | 新の置き場                                                                                          |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `initializers[].tensor`（safetensors のテンソルキー） | **initializer 名そのもの**になる（束縛表は `(グラフ名, initializer 名)` で引き、block id は内部名） |
| `storage.dtype`                                       | `binding[].encoding.codec`                                                                          |
| `storage.scale`（scale テンソルのキー）               | `binding[].encoding.scale.block`                                                                    |
| `storage.group_size`                                  | `binding[].encoding.groupSize`                                                                      |
| `initializers[].shared`（`{ tensor: <貸し手キー> }`） | `{ "shared": true }` — **借り手の名前 = 貸し手の initializer 名** MUST（鍵は名前）                  |

**名前を鍵にする理由**: v1 の initializer 名は torch.export の placeholder 名
（`p_model_layers_0_mlp_gate_proj_weight`）で、上流の鍵（FQN）は `tensor` 欄が別に持っていた。
`tensor` 欄を消すと、上流 checkpoint の取り込み（決定 19）と LoRA の対象解決（段 5）が鍵を失う。
名前を FQN にすれば表を 1 つも足さずに済む（実測: ミラー 77 グラフの名前・キーは全て ASCII で
最長 87 文字・衝突 0）。移行 CLI は v1 の `tensor` 欄で名前を付け替える（`values` とノードの
参照も同時に）。

狙いは「**1 アーキ・1 量子化方式・1 グラフ**」である。PTQ の席（格納 dtype / group /
session ノブ）と、同一 config の重み差し替え（fine-tune）から、グラフが独立する。
実測の裏付け: 調査した資産では、PTQ の席ごとのグラフ差分は `initializers[*].storage` **だけ**
だった（irodori dit f32/f16/i8/i4・anima f16/i8・sbv2 voice で実測）。

**QAT は別グラフのまま**である（`static_quantize` 485 ノードと焼き込み活性 scale 403 個を持ち、
`requires.ops` / `values` / `outputs` まで通常席と違う — グラフ自体が方式の一部）。

### 13.2 scale を rank 2 group 形へ統一し、`rowAxis` を宣言に出す

§6.1 のとおり。i8 の keepdim broadcast 形は廃止し、`conv_transpose1d` の i8 だけ
`rowAxis: 1` と宣言する。消費側 op から軸を**導く**のはやめるが、`weightChannelAxes`（`plan.ts`）は
**宣言との突合**として残す — 宣言と消費側の軸が食い違うと GPU 常駐経路が scale を別の軸に当てる
沈黙誤値になる（実装は `planWeightResidency`）。

### 13.3 合流層 — 規則は 1 箇所に置く

外出しは**可逆**でなければならない。重み取得前に**グラフ + 選択済み binding を合流**し、
合流後表現に対して:

- 格納の規則一式 — 量子化 codec は scale 必須 / `int4-sym-g` は `groupSize` 必須で 2 冪 ≥ 16 /
  `int2-off` は group 不可 / 行長の整除 / payload 長と block 長の突合 / piece の門（先頭次元の被覆・
  中間 piece に詰め物無し・`rowAxis != 0` は分割不可）
- `prepareContainer` の検査・常駐計画・見積り（合流後表現を実行グラフへ写す `mergedGraph` の後段）

を走らせる。**格納の規則の置き場は runtime の `format/container/` の中で 2 つに分かれる**。scale /
`groupSize` の有無は descriptor の読み手 `descriptor.ts`（`parseEncoding`）が見る。メモリ内容器
（`openMemoryContainer`）は入力形が違うので、同じ有無を `memory.ts` が見る。値域・行長の整除・i2 系の
宣言 shape・scale 長・payload 長・piece の門は `bind.ts` が見る — 2 文書に依存しない `bindDeclarations`
と、2 文書から呼ぶ外皮 `bindGraphs`。メモリ内容器も同じ `bindDeclarations` を通る。IR の宣言の読み手は `parseIrDeclaration` /
`parseIrDeclarationValue`（`packages/runtime/src/format/ir.ts`）の 1 種で、ホスト製のグラフも
`parseIrDeclarationValue` を通してからメモリ内容器へ渡す（`packages/runtime/mod.ts` の doc）。

**Python 側も合流後表現を既存検査へ渡し、規則を二重実装しない**
（`tools/exporter/src/karume/verify.py` の `bind_graphs` が TS `bindGraphs` の鏡像）。

### 13.4 変わらないもの

意味論 dtype（`f32` / `i32` / `bool`）・shape と次元言語・`states` スロット・`nodes` の形・
`requires.ops` の契約・op セット・「未知のトップレベルキーは fail loudly」。

### 13.5 検収

**段 1 の主検収**は 3 本である。

1. **分解 → 合流の往復**で、元の宣言・initializer の順序・常駐計画・見積りが一致する（CPU）。
2. **77 グラフ全部で、新旧の適格述語の結果が一致する**
   （`plan.ts` の `eligibleCompressedInitializers` / `i2EligibleInitializers` / `i4EligibleInitializers` — `executableOps` へ畳んだ後も同じ集合が出る）。
3. **同じグラフを再 export するとグラフ記述がバイト同一になる。** `krg` の同一性を内容ハッシュで
   判定する（§9）以上、`graphs[name]` の直列化まで決定的でなければならない。**IR v2 の直列化
   規則は [ir-v2.md](ir-v2.md) の「正準直列化」節で決め切った**（2026-09-22 — キー順はスキーマ順、
   名前キーの map は code point 順、数値の綴りは ECMAScript `Number::toString`、空白なし）。
   グラフ記述全体（`graphs` のキー順・`const` 目次・`capabilities` の集合）も同じ規則に従う。
   CPU 試作 ① はグラフ名だけを整列し、IR 本体は Python 側の serialize 順をそのまま持っていた —
   その状態では、抽出した `krg` と独立生成の `krg` のバイト同一は「同じ書き手で書いた場合」に
   しか成立しなかった。
