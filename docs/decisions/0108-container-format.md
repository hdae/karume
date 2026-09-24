# 0108: Karume 専用コンテナ形式 — `krm` / `krg`・block / part・codec 台帳

- Status: accepted（2026-09-22 — 段 0 の成果物として proposed で起票し、同日の段 1 着手の裁定で
  accepted。codec 登録名 4 種（決定 13）も同時に確定。本 ADR と [container-v1](../container-v1.md) が正本）
- Date: 2026-09-22
- 対象（段ごとに触る面の宣言 — 現時点では 1 行も実装していない）:
  - 仕様: [docs/container-v1.md](../container-v1.md)（新設 — 物理形式の正本）/
    [docs/ir-v2.md](../ir-v2.md) → IR v2 へ改訂（`storage` の外出し・`scale` の rank 2 統一）
  - runtime: `packages/runtime/src/format/`（`container.ts` / `ir.ts` / `safetensors.ts` /
    `i2.ts` / `i4.ts` / `i8.ts`）・`packages/runtime/src/runtime/`（`session-build.ts` /
    `plan.ts` / `weight-residency.ts` / `executor.ts`）
  - hub: `packages/hub/src/`（`manifest.ts` を `karume/5` へ・`fetch.ts` / `sources/` を
    block 単位取得へ）
  - models: 家族入口の `components` 席・`gemma/qat.ts` / `speculative.ts` / `ple-shard.ts`
  - exporter: `tools/exporter/src/karume/`（`emit.py` / `verify.py` / `repack.py` /
    `shards.py` / `quantize.py` / `modelcard.py`）・`tools/export-recipes/`・**移行 CLI**（新設）
  - 運用: `docs/release-runbook.md` / `hf-upload.zsh` / `karume dist` / 門番
    （`distribution_gate` / `assets_gate`）
- **Supersedes**（本 ADR が上書きする既存決定 — 既存 ADR の本文には追記で対応する）:
  - ADR [0003](0003-ir-v1.md) の**コンテナ規約**（`docs/ir-v2.md:86-91`「配布形は safetensors
    1 ファイル・`__metadata__.karume_ir` にグラフ JSON」）
  - ADR [0037](0037-karume-monorepo.md) **§3**（「独自拡張子 `.krm` は不採用」・
    「1 グラフ = 1 safetensors」）
  - ADR [0063](0063-safetensors-physical-layout.md)（safetensors 物理配置の契約 — 隙間なし・
    要素整列・固定書き出し順 `F32 → I32 → I4 → 偶数要素 F16 → 奇数要素 F16 → I8`）
  - ADR [0081](0081-shard-spec-v2.md) 決定 1 / 3（shard 0 はグラフ専用・常時分割）
  - ADR [0090](0090-shard-spec-v3-tensor-pieces.md) **決定 2**（受理上限 `SHARD_BYTE_LIMIT` =
    256 MiB 1 本・ファイル長で測る）
- 関連: ADR [0038](0038-manifest-v1.md) §3 / §4（実行既定は manifest 所有・配布者に runtime の
  綴りを書かせない — この分担は**変えない**）/ [0041](0041-manifest-v2.md) /
  [0071](0071-manifest-v3-shards.md) / [0075](0075-quant-presentation.md)（manifest `karume/4` —
  本 ADR の先に `karume/5` へ繰り上がる）/ [0069](0069-packed-w4-storage.md)（packed 4bit の
  値域・pack 順・group scale 形 — codec 台帳へ逐語移送）/ [0097](0097-gemma4-qat-integration.md)
  （INT2 格納と QAT の混成格納・SRQ）/ [0019](0019-i8-weight-execution.md)（i8 格納）/
  [0089](0089-memory-limits-preflight.md)（宣言から RAM を事前見積りする流儀）/
  [0105](0105-packed-static-quantize-activations.md)（packed int8 活性 — 低 bit codec の前提）/
  [0106](0106-device-keyed-references.md)（環境キー別 sha256 参照門 — 移行の検収で「緑にしない」
  規律が効く）/ [0058](0058-numerics-opt-in-contract.md)（数値 opt-in — 2 段 scale を将来入れる
  ときの席）/ [assets-layout](../assets-layout.md) / [release-runbook](../release-runbook.md)

## Context

配布形は今日「safetensors 1 ファイル + `__metadata__.karume_ir` にグラフ JSON」で、量子化格納
`I4` / `I2` は safetensors の**方言**である（公式ライブラリ 0.8.0 は拒否する —
`tools/exporter/src/karume/modelcard.py:368-376`・[limitations](../limitations.md):597-616 が
「次の manifest format 変更時に独自形式への移行を検討」と既に書いている）。つまり
「汎用 safetensors ツールで開ける」という当初の利得は、量子化席では**すでに失われている**。

失われていない利得のために払っている費用が、実物で 5 つ数えられる:

1. **グラフと重みの分離が物理配置のハック**になっている。shard 0 をデータ節 0 テンソルの
   「グラフ専用ファイル」にする規約（ADR 0081 決定 1）で分離を作っており、ミラー 447 本のうち
   `karume_ir` を持つのは 77 本・すべてデータ節 0 テンソル（実測）。IR の総量 14.2 MB は
   データ節 73.8 GB の **0.019 %** で、分離のために本数を増やしている割に中身は小さい。
2. **テンソル名に物理の都合が綴られている**。分割テンソルは `<名前>#00002-of-00003` という
   piece キー（`packages/runtime/src/format/container.ts:168`）で、名前空間に配置が漏れている。
3. **付随資産が物理配置に依存した区間読みになっている**。PLE sidecar は safetensors の
   `byteOffset` を自分で取り出して 1 token 分 9,100 B を 2 回に分けて読む
   （`packages/models/src/gemma/ple-shard.ts:320-337`）。
4. **末尾ゼロ詰めが読み手側に散っている**。`writeBuffer` が 4 の倍数でないサイズを拒むため、
   転送側に「丸ごと / piece 列の末尾だけ詰める」分岐が 3 つある
   （`packages/runtime/src/runtime/session-build.ts:819, 856, 883`）。詰め物は**バイト列では
   なく実行時の都合**なのに、読み手が毎回作り直している。
5. **取得単位 = ファイル 1 本**で、ホスト RAM のピークが「定数 + 最大 shard 1 本」でしか
   抑えられない（ADR 0090 決定 2）。hub は HTTP Range を発行せず、区間読みは温め済み
   キャッシュの中だけである（`packages/hub/src/sources/hf.ts:81-87, 153, 194-236`）。

一方で、やりたいことが 3 つ増えた。**無改変の上流 safetensors を読む**（対応済みアーキに
上流重みを持ち込む）・**LoRA をその場で載せる**・**部品単位で差し替える**（transformer だけ
別出所）。この 3 つはいずれも「グラフ契約」と「重み供給」を**別の物として組み合わせる**形を
要求するが、今日の形式は 1 ファイルの中で両者が癒着している。

加えて量子化側からの要求がある。gemma4 e2b の i4 g32 を三値 2 bit に落とすと格納は **−40 %**、
f16 scale 化だけで **−10 %**、group 32 → 128 で **−15 %**（実測ヘッダからの試算）。ところが
今日の宣言語彙は `storage.dtype` の**列挙**（`f32 | f16 | bf16 | i8 | i4 | i2 | i32` —
`packages/runtime/src/format/ir.ts:18`）なので、新しい詰め方は語彙を 1 つ増やすたびに宣言・
検査・展開・実行の全層へ枝を生やす。実測で、**宣言を足すのは 3 ファイル**（`ir.ts` /
`verify.py` / `safetensors.ts` の bit・整列表）で済むのに、**実行を足すと非テスト 22 ファイル・
WGSL スナップショット変種 10〜21 本**になる（`wi2` 10 本 / `wi8` 21 本 / `wi4` 21 本 —
`packages/runtime/tests/fixtures/wgsl/` の実数）。

## Decision

### 1. 専用コンテナを 2 種に分け、物理形式は 1 つにする

- **`krm`**（model）= グラフ + 学習済み重み + scale + 資産。
- **`krg`**（graph）= グラフ + **重み block を持たない**もの。定義は「重み block を持たない」
  であって「重み非依存」ではない — QAT は `static_quantize` の attr に活性 scale を焼き込む
  （ADR 0097 追記 2）ので、`krg` でも重みに由来する数を抱えることがある。
- 物理形式・グラフ表現・descriptor のスキーマは**両者で完全に同一**。区別は**magic 4 B の 1 文字**
  （`KRMC` / `KRGC`）だけが持ち、descriptor の中に `kind` 欄は置かない — 同じ事実を 2 か所に
  持たせない（横断の不変条件「導出できる状態を別欄で持たない」）。JSON を 1 バイトも読む前に
  種別で弾けるという副次利得もある。
- `krg` は**重みの束縛表**を持たない（重みの要求は `values` の宣言 shape / dtype から導出する）。
  ただし**const 領域の block を initializer へ結ぶ表**（`const.constants[]` =
  `{ graph, initializer, block, encoding }`）と `const.length` は**グラフ記述**が持つ。これが
  `krg` 単独で `const.*` を供給できる唯一の手段であり、`krg` が part 1 の寸法を知る唯一の手段
  でもある（CPU 試作 ① の指摘 — 「束縛表を一切持たない」と読むと const の `encoding` をどこから
  読むのかが決まらない）。provenance は引き継がない。持ち出せる資産は `rope_base` のような
  **重み非依存**のものだけ。
- `@karume/models` への `krg` 同梱は**形式として許すが標準配布にはしない**。全グラフの重複除去
  後の総量 9,444,431 B（9.0 MiB）に対し `@karume/models` の公開物は 1,408,706 B（1.34 MiB）で、
  同梱は **6.7 倍**になる（却下案 2）。

### 2. ヘッダと **2 文書 descriptor**

```text
[magic 4 B]   "KRMC"（krm）/ "KRGC"（krg）
[u32 LE container version]
[u64 LE graph descriptor length]
[u64 LE model descriptor length]     ※ krg では 0 MUST
```

descriptor を**グラフ記述**と**モデル記述**の 2 文書に割る。

| 文書       | 持つもの                                                      |
| ---------- | ------------------------------------------------------------- |
| グラフ記述 | `graphs`（IR v2）・**const 領域の目次**・`capabilities`       |
| モデル記述 | **束縛表**・重み / 資産の目次・parts の内部配置・`provenance` |

割る理由は §4 の「`krg` をバイトコピーで抜ける」を成立させるためである。1 文書のままだと
`krg` を作るたびに JSON を再生成することになり、「同じグラフから抜いた `krg` が生成器の版で
バイト違いになる」経路が残る。

descriptor は `session` / `gpuFeatures` / `requiredLimits` / `label` / `description` を
**持たない**（ADR 0038 §3 — 実行既定は manifest の所有で、配布者に runtime の綴りを書かせない。
`fromContainer` では呼び手が渡す）。

### 3. 物理配置 — 単一形と分割形は同じ並び

分割形:

| part   | 中身                                                                     |
| ------ | ------------------------------------------------------------------------ |
| 0      | `[ヘッダ][グラフ記述][モデル記述]`                                       |
| 1      | **const 領域**（グラフの所有・block の offset は**領域先頭からの相対**） |
| 2 以降 | 重み / 資産の block                                                      |

単一形は**同じ順で連結**する（境界は 64 B 整列）。つまり単一形と分割形はバイト列として
「切れ目があるか否か」だけが違い、descriptor の読み方は 1 本で済む。

const を専用 part に隔離するのは、const が小さいとは限らないからである（実測: gemma4 e2b i4 で
0.08 % だが、birefnet 2048 f32 で 29.3 MB = 2.5 %・sbv2 F1 front i4 で 2.1 MB = **29.7 %**）。
準備時に取るのは part 0 だけで、const は要るときに取る。

part 0 だけで admission（グラフの受理・実行計画・見積り）が閉じる。これが「**先にグラフだけ
取って `prepareModel` → admission → 重み**」の順序を現行より軽く保つ条件である。

### 4. `krg` は**バイトコピー**で抜ける

`krg` = `[ヘッダ'][グラフ記述][const 領域]` の連結。**offset の書き換えを 1 バイトも伴わない**
（const block の offset が「const 領域の先頭からの相対」だからである — 決定 3）。

これで「`krm` から抜いた `krg`」と「最初から `krg` として書いた同じグラフ」がバイト同一になり、
`krg` の同一性を**内容ハッシュ**で判定できる（名前でも「同アーキ」でもなく）。const の束縛表を
グラフ記述が持つ（決定 1）ので、抜いた `krg` だけで const の供給が閉じる。

バイト同一の前提がもう 1 つある: **再 export 間でもグラフ記述がバイト同一であること**。これには
IR v2 の**直列化規則**（キー順・数値の綴り）を決め切る必要があり、段 1 の作業項目に入れる
（CPU 試作 ① はグラフ名だけ整列し、IR 本体は Python 側の serialize 順をそのまま持っていた）。

NOTE: 設計案 v2 はこの流儀の先例として `distribution.py:570` の「RoPE 表の共有」を挙げていたが、
**その参照は現物と合わない** — gemma4 の RoPE cos / sin 表は既に配布物から外れており、ホスト側が
`pipelineConfig.rope` の宣言から実行時に組む（`tools/export-recipes/gemma4/distribution.py:35`）。
先例としては数えず、本 ADR は内容ハッシュ判定を新規の決定として置く。

### 5. **block** = 取得・検証・解放の単位・上限 **32 MiB 以下**（独立定数）

- block は独立に取得し、独立に sha256 を検証し、GPU へ上げたら独立に解放できる。
- block は **part をまたがない**。
- **上限は 32 MiB 以下**（`length ≤ 33,554,432 B`。「未満」ではない）。これは part 長からの
  派生ではなく**独立した定数**である。根拠は 3 点（CPU 試作 ① / ② の実測・2026-09-22）:
  1. **割っても digest の総費用は増えない**。256 MiB を一括 digest すると 923 MiB/s、
     32 MiB × 8 に分けると 938 MiB/s（比 0.984）である。「block 単位で検証する」設計に
     digest 側の言い訳は要らない。
  2. **anima transformer の最大重み（2048×8192 f16）がちょうど 33,554,432 B で、56 本がこの値に
     乗る**。上限を「以下」にすれば piece 分割は 0 本、「未満」にすると同じ 56 本が分割される。
     境界の定義 1 つで実資産の分割件数が 0 と 56 に割れる。
  3. **const block は piece 分割の機構を持てない**（piece 列を表せるのは重みの束縛表だけで、
     `krg` はそれを持たない — 決定 1）ので、const 1 本が上限に収まらなければ表現できない。
     現状の最大は birefnet 2048 の f32 定数 **29.3 MB** で、32 MiB まで約 2.7 MB の余裕しかない。
- digest には **`subarray` を渡す**（`slice` コピーを挟むと 943 → 641 MiB/s）。
- 並行 digest は効く（16 MiB × 16 を `Promise.all` で **3,702 MiB/s** — 逐次 16 MiB の 2.07 倍・
  256 MiB 一括の 4.0 倍）。一括 digest ではこれが取れない。
- ブラウザは未測。Chrome は digest の入力を Blink 内部へ全量コピーする
  （`packages/hub/src/fetch.ts:68-70` の記録）ので、block 化はブラウザでこそ効くはずである
  （**推測** — コピー量が block 長で頭打ちになる）。
- **将来の見直し（16 MiB へ下げる）**: 交互計測では 16 MiB = 1,784 MiB/s に対し 32 MiB =
  938 MiB/s で、速さだけを見れば 16 MiB が有利である。下げるときの副作用は 2 つ —
  ①**const の上限が先に壊れる**（birefnet の 29.3 MB が入らなくなり、上の根拠 3 の逃げ道
  〈仕様 §10 の未決〉が必要になる）②**block 件数と descriptor が倍に膨らむ**。切り替えは定数
  1 つなので、段 3 の RAM ピーク実測まで持ち越す。
- **撤回**: 草案は「32 MiB ちょうどまで 1,865 MiB/s・33 MiB から 948 MiB/s と半減する」という
  崖を根拠にしていたが、再実測で**再現しなかった**。あの数はサイズを昇順に連続計測したときだけ
  出る測り方の産物で、16 MiB と交互に回すと 16 MiB = 1,784・32 MiB = 938・40 MiB = 959・
  64 MiB = 933 MiB/s となり、**崖は 32 MiB の手前にある**（**推測**: 32 MiB は glibc の mmap
  閾値の上限と同値で、この境界から先は確保のたびに mmap / munmap が走る）。サイズを振る digest
  計測は交互に回さないと結論が逆になる。

### 6. **part** = 配信粒度・長さは**モデル規模で書き手が選ぶ**

- part の長さは書き手の選択 **`{256, 512, 768, 1024} MiB`**（既定 **256 MiB**）。
- manifest と descriptor の**両方**が各 part の長さを宣言し、読み手は宣言だけで host RAM を
  事前見積りできる（ADR 0089 の流儀 — 1 バイトも取る前に器の寸法が決まる）。
- **exporter の既定を 256 から動かすのは段 3（block 化）の検収後**。段 1 / 2 では取得単位が
  まだ part なので、1024 MiB を選ぶとピークが **+768 MiB** 乗る。
- ADR 0090 決定 2 の根拠 3 点のうち、R1（RAM ピーク）と R3（Chromium の単一 ArrayBuffer 上限
  2,145,386,496 B）は**block 上限へ移る**。part に残るのは R2（ファイル数・リクエスト数が
  hub の 4 並列と `MAX_SHARDS` = 1024 の内側）だけである。
- 実資産の実態: 447 本すべて 256 MiB 以下・最大 254.3 MiB・中央値 224.0 MiB・合計 68.8 GiB。
  `assets` / `extras` に 256 MiB 上限が掛からない現行の免除（`packages/hub/src/manifest.ts:76-79`）は
  **今日 1 本も使われていない**（PLE sidecar も自分で 9 / 5 本に割れている）。

### 7. 末尾 padding は**書き手が焼く**

`writeBuffer` は 4 の倍数でないサイズを validation で拒む。今日はこれを読み手が毎回作って
いるが（決定の Context 4）、**書き手が block 長を 4 の倍数に揃えて焼く**。

- block 先頭は 64 B 整列（∴ offset は自動的に 4 の倍数）、**block 長は 4 の倍数**。
- 詰め物のバイト値は **0x00 固定 MUST**。覆うハッシュは**2 段構え**である — **block 末尾の
  詰め物は block の sha256 の対象内**（書き手が焼くので決定的・再梱包のビット同一が詰め物を
  含めて定義される）、**block 間 / part 間の詰め物は part の sha256 だけが覆う**。
- 詰め物を掛けてよいのは**丸ごと 1 本の block と piece 列の末尾**だけである。**中間 piece には
  掛けない**（掛けると次の piece の先頭を潰す — `session-build.ts:816-819` が今そう書いている）。
  中間 piece のバイト長は**元から 4 の倍数 MUST** で、書き手は行の刻みを `4/gcd(rowBytes,4)` に
  丸めて切り、丸め切れないときは fail loudly。
- これで `session-build.ts:819, 856, 883` の分岐 3 つと、**ADR 0063 の書き出し順規約が退役する**
  （順序で整列を作る必要が無くなる）。
- 束縛規則②（旧「末尾でない piece の block 長は 4 の倍数」）は「**全 block の長さが 4 の倍数**」へ
  一般化する。**const と資産も含む全 block**に掛かり、長さが 4 の倍数でないもの（bool 表・
  奇数長の f16 など）は詰め物込みで宣言する。**消費側は宣言 shape から論理長を導く**。

### 8. 完全性 — ハッシュの役割を 3 つに分ける

- **descriptor は外側の期待 hash + 長さで先に検証する**（manifest の FileRef、または
  `fromContainer` の呼び手が渡す pin）。descriptor 自身は自分の正しさを証明できない。
- **実行時の完全性は descriptor と block の sha256**。cold は block ごとに一括 digest、warm は
  記録ハッシュの文字列比較だけで **digest 0 回**（キャッシュヒット時に GB 級の digest を
  走らせない — `packages/hub/src/fetch.ts:66-73` の現行規律を継承）。
- **ファイル全体の sha256 は公開・再梱包の突合用**として分離する（実行時には使わない）。
- ハッシュの役割 3 分離を仕様に明記する: ①**取得物の期待ハッシュ**（外側が持つ）②**派生
  キャッシュキー**（入力 digest + recipe 版 + 変換設定 — 生成前に決まる）③**派生物の内容
  ハッシュ**（生成後に計算する）。この 3 つを 1 つの欄で兼ねると、上流重みの取り込みで
  「まだ存在しないものの内容ハッシュ」を要求する形になる。

### 9. GPU 転送とメモリ契約

- **`mappedAtCreation` は採らない**（却下案 6）。全 block が揃うまで unmap できず、
  「block ごとに解放する」という目的と逆を向く。`queue.writeBuffer(dstOffset)` を block ごとに
  呼ぶ。
- wgpu（Deno）の `writeBuffer` staging は **submit 完了まで解放されない**（実測: `createBuffer`
  2 GiB 後 2,311 MiB → `writeBuffer` 後 4,359 MiB —
  `docs/research/2026-08-08-vram-oom-misreport.md:73`）。したがって現行の
  「shard ごとに空 submit + `onSubmittedWorkDone`」（ADR 0070 決定 3）は **part 単位（または
  同時処理バイトの予算単位）のフェンス**として保つ。
- **errorScope の粒度は block・フェンスの粒度は part** に分ける（2026-09-22 実測・Arc B570 /
  Deno 2.9.6・256 MiB・5 回の中央値・`.claude/reviews/2026-09-22_codex-format-design/spikes/errorscope/`）:
  push/pop 2 本組は **1.81 µs / 回**でほぼ無料、高いのは空 submit + `onSubmittedWorkDone` の
  **13.0 ms / 回**である。pop もフェンスも block ごと（32 MiB × 8）だと 127.4 ms（現行相当 73.8 ms
  の 1.73 倍）、pop だけ block ごとでフェンスは 256 MiB ごと 1 回だと **56.5 ms（0.77 倍）**。
  失敗の帰属粒度を block へ上げつつ壁時計は悪化しない形が実在するので、決定 9 の宿題は閉じる。
- 重ね合わせは「転送中 1 + 受信・検証中 1」のように**本数と合計バイトの両方**で制限する。
  展開（1 bit → f32 は 32 倍）も処理単位を分ける。
- 上限は**個別に持つ**（派生させない）: descriptor の長さ / 入れ子深さ / block 件数 /
  part 長の集合 / block 長 / 同時処理バイト / 展開領域 / GPU の device 上限。
- **`fromContainer(bytes)`（全量 ArrayBuffer の口）は残す**。Chromium の 2,145,386,496 B は
  **この口にだけ残る制限**であることを仕様に明記する（HF 公式配布は分割形なので当たらない）。
- **追記（2026-09-24・段 3e）— CPU 側の解放はフェンスを待たない**: 上のフェンスが律するのは wgpu の
  staging だけで、CPU 側のバイト列は `writeBuffer` が戻った時点で手放してよい（WebGPU 仕様: content
  timeline で呼び出しの時点に `dataContents` を写す）。Session 構築は part の block を 1 本ずつ読んでは
  上げて手放す（`WeightBatch.items` の lazy 化）ので、JS 側で参照が生きる重みのバイト列は part 1 本
  ぶんから item 1 本ぶん（block 1 本 + 同乗 scale + 展開席の f32 展開結果。展開席の piece 列の間は
  持ち越し scale の写し〈その initializer の scale 全量〉がもう 1 本生きる — 数え方の正本は container-v1
  §11）になる。ただし scan 型の
  取得元では block が hub の保持枠の view なので、上限は保持枠 1 本 + GC を待つ前の器のまま（container-v1
  §11）。段 3e の実測では scan 型の external 最大が最大 part の約 1.5〜4.4 本だった（追記 5 の 5）。フェンスは part ごと
  1 回のまま。実機での成立は `packages/runtime/tests/gpu_write_buffer_copy_test.ts`（戻った直後に
  毒で埋めても出力がビット一致）で固定し、ADR 0070 決定 3 の「フェンス後解放」は同日の追記で改めた。

### 10. `encoding` の宣言語彙 — bit 数は派生値にする

束縛表の各エントリが持つ供給記述:

```jsonc
"encoding": {
  "codec": "<台帳の登録名>",
  "packing": { "blockElements": 8, "blockBytes": 4, "alignBytes": 4 },
  "rowAxis": 0,
  "groupSize": 32,
  "scale": { "block": "<block id>", "dtype": "f32" },
  "zeroPoint": { "block": "<block id>" }
}
```

- **bit 数は `blockElements` / `blockBytes` からの派生値**にする。`bits` を宣言に置くと
  非整数 bpw（ビット面分解や外部形式の block 量子化）が表せず、payload バイト長も決まらない。
  現行は `numel × bits / 8` の厳密一致（`packages/runtime/src/format/safetensors.ts:139-152`）
  なので、ここを `packing` 経由にするのは**受理集合を変えずに一般化する**変更である。
- **整列要求は codec の属性**（現行 `DTYPE_ALIGN`: I8 = 1 / I4 = 4 / I2 = 4 —
  `safetensors.ts:43-53`）。GPU が `array<u32>` で束縛するかどうかで決まるので、
  宣言側から導けない。
- `groupSize` が行長に等しいとき = per-channel。

### 11. `scale` は rank 2 group 形に一本化し、`rowAxis` を宣言に出す（**IR v2 に含める**）

今日、scale の形は **2 種類**ある（i8 の keepdim broadcast 形 vs i4 / i2 の rank 2 group 形 —
`packages/runtime/src/format/i4.ts:56-58` が「受理集合が交わらない**別物**」と明記している）。
さらに i8 の**チャネル軸は宣言に書かれておらず、消費側 op から導いている**
（`packages/runtime/src/runtime/plan.ts:411-435` の `weightChannelAxes`。消費 op が食い違うと
`plan.ts:424-428` で落ちる）。

- i8 の per-channel scale は「行 = 先頭次元・group 長 = 行長」とみなすと rank 2 group 形
  `[shape[0], 1]` と**同じもの**になる（i2 は既にその形 `[N,1]` を採っている）。
- 唯一の例外は **`conv_transpose1d` の i8**（重み `[Cin,Cout,K]` でチャネル軸が 1）。これは
  `rowAxis: 1` と**宣言する**。
- これで「宣言だけで scale の意味が閉じる」。`weightChannelAxes` は消える。
- **段 1 の IR v2 に含める**。検収 = **77 グラフ全部で新旧の適格述語**
  （`plan.ts:451` / `:496` / `:519`）**の結果が一致**すること。

### 12. codec は**リポ内の不変な台帳**にする

- import 時登録は**しない**（横断の不変条件「全モジュール副作用ゼロ」に反する）。台帳は
  リポ内のデータで、エントリは
  `{ name, packing, levels, scale, zeroPoint, decodeCpu, executableOps, wgsl? }`。
- **3 つの軸を分けて報告する**: ①宣言として読めるか ②実行できるか ③どの op で圧縮のまま
  常駐できるか。この 3 分離は**現行が既にそうなっている**（`bf16` は①のみ — 宣言は valid で
  `RUNTIME_SUPPORT.storage` に無いので実行は fail loudly、`packages/runtime/src/ops/contracts.ts:760`。
  i2 は①②③だが③は linear / embedding だけ）。**新 codec を「宣言だけ先に受理する」道が
  既に通っている**。
- **未知 codec は重み取得前に拒否する**（現行 `asStorageDtype`（`ir.ts:271-277`）が語彙外を
  拒むのと同じ形）。
- 台帳の値打ちは「codec ごとに述語が生えるのを止める」ことにある。現行は
  `eligibleCompressedInitializers` / `i2EligibleInitializers` / `i4EligibleInitializers` の
  3 本で、i2 を足したときに i4 の述語がほぼ写された（`plan.ts:496-511` と `:519-534` が同文）。
  これを `executableOps` の 1 欄へ畳む。

### 13. 初版の codec は 4 種 — i8 / i4 / i2 / **三値**

既存 3 種の packing・levels・scale・復元式は
[container-v1 §6](../container-v1.md) へ**逐語で移送する**（値を 1 つも作らない）。

4 種めの**三値**（ternary）は:

- 値域 `{−1, 0, +1}` は **i2 の値域 `[−2, +1]` の部分集合**なので、**詰め方も復元も i2 と
  完全に同一**（`u = q + 2` ∈ `{1,2,3}`・コード 0 は未使用・復元は `fround((u − 2) · s)`）。
- したがって **runtime の追加は 0 行**。WGSL も CPU 展開も i2 のものをそのまま使う。
- exporter には **absmean 量子化器 1 本**を足す。
- それでも**別名の codec として宣言する**。理由は「資産を識別できるようにする」ため — 逆は
  成立しない。実資産の i2 は三値では**ない**（gemma4-qat e2b の I2 テンソル先頭 4 MiB を
  2bit コードで数えた実測で、`q = −2` が **5.8〜7.4 %** 出ている）。i2 資産を三値として
  読み替えると全要素の 6 % 前後が別の値になる。

### 14. 奇数 bit（3 / 5 / 6 / 7）は**ビット面分解を予約するだけ**

- 方式: 32 要素を `bits` 本の u32 へ「平面」で置く（語 b のビット j = 要素 j の第 b ビット）。
  **格納は厳密に `bits` / 要素**（捨てビット 0）で、平坦添字は常に 2 冪（32 要素 / 面）のまま
  保たれる。
- 予約するのは**台帳のエントリ形だけ**で、実装は需要が出た幅から行う。
- **「u32 語に n 個で余りビットを捨てる」は採らない**（却下案 5）。
- **base-243（GGUF TQ1_0 の 5 trit / byte）は採らない**（却下案 4）。

### 15. 低 bit 化は**速度の理由にしない**

decode は本機で帯域律速に**なっていない**（帯域利用率 11.5 / 25.5 % の実測）。低 bit codec を
速度目的で入れると、律速でない側を削ることになる。低 bit codec は **packed int8 活性
（ADR 0105）が前提**であり、効くのは**メモリ**である（決定の Context 末尾の −40 % / −10 % /
−15 %）。

### 16. exporter 側の一般化

- RTN / GPTQ の **bit 幅依存は 3 箇所（`max_level`）** — ここを引数化する。
- 三値の absmean は**冪等性の論証を本 ADR の追記として書き直す**（i8 / i4 の
  「amax 要素が厳密復元され fake-quant が不動点」の論証をそのまま流用できない）。
- **codebook 系は別裁定**（2 段 scale は「復元の丸めは f32 乗算 1 回」というビット一致の根拠
  （`packages/runtime/src/format/i4.ts:9-11`）を壊すので、ADR 0058 の数値 opt-in の席になる）。

### 17. IR v2 = IR v1 − `storage`（+ 決定 11）

- `initializers[name]` から `storage` を外して**束縛表**へ移す。狙いは
  「**1 アーキ・1 量子化方式・1 グラフ**」で、PTQ の席（格納 dtype / group / session ノブ）と
  同一 config の重み差し替え（fine-tune）からグラフを独立させることである。
- **QAT はグラフ自体が方式の一部なので別グラフのまま**（`static_quantize` 485 ノードと
  焼き込み活性 scale 403 個を持ち、`requires.ops` / `values` / `outputs` まで通常席と違う）。
- 合流層は薄くない。**重み取得前にグラフ + 選択済み binding を合流し、現行 `parseIrGraph` の
  storage 規則一式**（i8 / i4 は scale 必須・i4 の group_size は 2 冪 ≥ 16・i2 は group 不可・
  行長の整除 — `ir.ts:296-345, 685-720`）**と `prepareModel` の検査・常駐計画・見積りを
  合流後表現に対して走らせる**。置き場は runtime `format/` の 1 箇所。
- Python 側も合流後表現を既存検査へ渡し、**規則を二重実装しない**。

### 18. 旧 safetensors 形式との互換は**全部切る**

- 新パッケージは `karume/5` と新コンテナ**だけ**を読む。**両読みは実装しない**（却下案 3）。
- 旧形式を読む処理は**移行 CLI**（Python）に限定する: 旧単一形 / 旧 shard 列 / 旧 manifest →
  新 `krm` / `krg`。**旧入力は保持する**（消さない）。
- **上流 checkpoint 取り込み用の safetensors reader は別用途として保持**する（無改変の上流
  重みを読む経路 — こちらは配布形ではない）。
- 旧版パッケージは旧 revision（40 桁 SHA）を pin しているので**動き続ける**。HF リポの
  **履歴は残す**（release-runbook の「リポ削除・再作成」は使わない）。
- 移行の対象（実数）: `models/` ミラー **11 本 447 ファイル 69 GB** の再梱包 /
  golden **32 モデル 82 ファイル** + gemma4-ple-packed **8 本**の再 export（生成器あり・CPU・
  固定 seed）/ `static-quantize-oracle` / examples のローカル経路 / テストヘルパ
  `shard-files.ts`（呼び手 **33**）と `safetensors-write.ts`（**7**）/ 門番（`assets_gate` は
  ディレクトリ・`distribution_gate` は manifest の存在しか見ていない → **中身を見る**形へ）/
  `hf-upload.zsh` の glob / release-runbook / exporter の emit・verify・repack・dist・modelcard
  （`I4` / `I2` 方言の節を退役）。
- 再アップロードは **pin のある 10 リポ**（gemma4-qat は未公開 → ミラー再梱包のみ）。固定順序が
  要るのは **anima → SHA 確定 → anima-extra** だけ（越境参照 4 鎖のため）。
- `outputs/series/` は大掃除する（probe・重複変種は即削除。レーンが参照する系列は段 1 後に
  新形式で再生成）。

### 19. 部品単位の差し替え

- 系列入口に `fromPretrained(source, { components: { <役割>: ComponentSource } })` を足す。
  継ぎ目は既存の **`ComponentOpener`**（`packages/models/src/hub/components.ts:89` —
  `AnimaPipeline.#build` は既に `open: ComponentOpener` を引数で受けている
  （`packages/models/src/anima/pipeline.ts:723-726`））。
- 検査（別出所の重みとグラフ宣言の突合・quant 席と実行設定の整合・`pipelineConfig` の上書き）は
  **admission に置き、「重みを 1 バイトも取る前」を保つ**。
- **runtime は出所を知らない**。hub は取得元を N 本扱えるようにするだけ。
- 推定してよいのは「どの部品か（キー指紋）・変換表の版・格納 dtype」の 3 つだけ。**宣言必須**は
  「ベースの `krg`・`pipelineConfig`・quant 席・上位集合のときの除外」。
- **diffusers の `strict=False` は採らない**。不足も余剰も**全件列挙で拒否**する。単一ファイル
  checkpoint は「部品名 → 重み供給」の集合として扱い、未選択部品は**理由付きの除外**として
  宣言する。
- 実測の裏付け（CPU 試作 ②・variant 2 本 = anima-wai-v1.0 / anima-copycat-20260610）: anima
  transformer の recipe は **置換 23 件 + 削除 8 件**（diffusers 0.39.0 の Cosmos 2.0 表）
  **+ 前置規則 3 件・数値変換 0**（IR キーへは `model.` を前置）である。checkpoint **685 本**が
  **DiT 567 + conditioner 118** に 1:1 で割れ、**不足 0・余剰 0・形違い 0** — reshape /
  transpose / 連結・分割は 1 本も要らない。
- **`llm_adapter.` 前置の 118 本は「余剰」ではなく、`routedTo` で宣言済みの振り分けとして書く**
  （text_conditioner 側へ回る）。
- **置換表は順序が意味を持つので配列で持つ**。`adaln_modulation_self_attn.1` を `self_attn` より
  先に当てないと対応が変わるため、JSON object にすると順序契約が消える。
- 上流は bf16 なので **bf16 デコーダが前提**になる（`RUNTIME_SUPPORT.storage` に bf16 は
  無い — `ops/contracts.ts:760`）。**bf16 → f32 厳密 → `Math.f16round`** の順で丸めると配布形の
  f16 shard と **567/567 バイト一致**する。f32 格納の 113 本も**値は既に f16 へ丸め済み**である
  （`tools/export-recipes/anima/export.py:931` の `round_weights_to_f16` がラッパ全体を回し、
  格納 dtype は emit 側が別に決めるため）。「f32 席 = 素の bf16 → f32」と仮定すると 113 本が全部不一致に見える。
- 重み recipe は**宣言的なデータ**（版付き・Python / TS 双方が読む）にする。**任意の前処理を
  書ける言語にはしない**（却下案 7）。
- **未決**: 上流の表が動いたことを検出する手段が無い（実行時に Python は居ない）。`krg` に
  「対応済み上流の鍵集合ハッシュ」を持たせるか、派生キャッシュキーの recipe 版だけで足りるかは
  実需が出たときの裁定とする。

### 20. manifest `karume/5` との責務分担

| 所有者              | 正本                                                                                                                                                                                                                                                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| manifest `karume/5` | model / quant の一覧・既定選択・`label` / `description`・`session` 既定・`gpuFeatures`・`requiredLimits`・**入口の FileRef**（`quants[].container = { descriptor: FileRef, parts: FileRef[] }`・共有 `krg` の `graph: FileRef`）・越境参照（容器単位 = descriptor と全 part が同一 repo + revision）・`pipeline` / `pipelineConfig` |
| `krm` の descriptor | `graphs`（または graph の内容参照）・束縛表・blocks・parts の内部配置・assets・provenance・capabilities                                                                                                                                                                                                                             |
| `krg`               | 計算契約・重み非依存の定数とその依存条件（config 値）                                                                                                                                                                                                                                                                               |
| models              | `pipelineConfig` の意味検査・実行設定の明示指定優先・assets の解釈                                                                                                                                                                                                                                                                  |
| 配布成果物          | LICENSE / NOTICE の本文（リポ直下 — ADR 0071）。単一 `krm` の持ち出し用に descriptor の `provenance` へ**ライセンス識別子と NOTICE 参照**を載せる（本文は載せない）                                                                                                                                                                 |

hub は 1 バイトも取る前にファイルごとの `size` / `sha256` が要る（キャッシュキー・in-flight
予算・器の寸法）ので、**parts の FileRef は manifest に残す**（descriptor の中だけに置くと
descriptor を取るのが 2 周目になる）。

## 検討した代替案

1. **単一 `krm` だけにする（`krg` を作らない）** — 却下。上流重み・LoRA・部品差し替えの 3 つは
   いずれも「グラフ契約」と「重み供給」を別々に組む形を要求する。`krm` に差し替え口を足すと
   実質 `krg` になり、しかも「グラフだけを配る」ときに重み用の欄を空にした `krm` を配ることに
   なって、受理集合が「空の欄」で表現される（fail loudly が弱まる）。
2. **`@karume/models` にグラフを同梱するのを標準にする** — 却下。既知モデルは楽になるが、
   **パッケージと資産の版が結合する**（グラフの更新 = JSR 公開・未知モデルはパッケージ更新
   待ち）。サイズも 1.34 MiB の公開物に対し 9.0 MiB で 6.7 倍。形式として同梱を**禁じはしない**
   （決定 1）が、標準にはしない。
3. **移行版で旧形式と新形式の両読みを維持する** — 却下。草稿 v1 と設計案 v2 §9 はこれを推して
   いたが、**撤回する**。pin が 40 桁 revision なので旧版パッケージは動き続け、両読みが守るのは
   「新版パッケージで旧資産を読む」場合だけである。その用途は**移行 CLI が 1 度だけ実行する
   変換**で足り、両読みを本体に置くと、旧形式の検査・piece キー・物理配置ハック・`karume_ir`
   の逐語保存という**退役させたい規約一式が読み手に残り続ける**。旧形式は magic で分岐できる
   ので、必要になったら移行 CLI 側で読める。
4. **base-243（GGUF TQ1_0 相当・5 trit / byte）で三値を 1.6875 bpw にする** — 却下。2 bit 詰めに
   対して **−15.6 % しか縮まない**一方で失うものが大きい。5 要素粒度は 2 冪でないので
   「平坦添字から語内位置をシフトで割る」不変条件と「行頭が語境界に来る」不変条件が**両方
   壊れる**。行長 `K` に `% 5` の整除条件が要り、gemma4 の実形（1536 / 2048 / 6144 / 12288）は
   **どれも 5 で割れない**。さらに base-243 の桁取り出しは要素ごとに乗算が要り、決定 15 の
   「律速は ALU 側」という実測に照らして悪化方向である。
5. **u32 語に n 個詰めて余りビットを捨てる**（3 bit なら 10 個 + 2 bit 捨て） — 却下。実効
   bit / 要素が 3.200 / 5.333 / 6.400 / 8.000 と膨らむ（3 / 5 / 6 bit で +6.7 %・7 bit で
   +14.3 %）うえ、**1 語あたりの要素数が 2 冪でなくなる**。GEMV 族は語ごとに完全展開している
   のでループ内に除算は出ないが、**scale の group 添字**（`packages/runtime/src/kernels/linear-gemv.ts:404-406`
   の `(unit·刻み) >> shift`）が 2 冪前提で書かれており、シフトが乗除算に変わる。
   i4 の `i4GroupShift`（`kernels/weight-storage.ts:57-71`）が「group は 2 冪」を要求している
   のも同じ理由である。
6. **`mappedAtCreation` で staging コピーを減らす** — 却下。目的と逆を向く。マップした
   バッファは**全 block が揃って unmap するまで解放できない**ので、「block ごとに取得・検証・
   解放する」という本 ADR の核が成立しなくなる。なお `mappedAtCreation` はリポジトリ全体で
   **1 箇所も使われていない**（重みも入力も常駐テンソルも `queue.writeBuffer` 1 本）。
7. **重み recipe を「任意の前処理を書ける宣言的言語」にする** — 却下。上流 FQN ↔ IR キーの
   写像に必要なのは有限個の演算（slice / split / transpose / alias / 定数供給 / 理由付き除外）
   で、実測でも anima transformer は**置換 23 件 + 削除 8 件 + 前置規則 3 件・数値変換 0**で
   ある（決定 19）。任意の言語にすると ①Python と TS の 2 実装で意味が一致する保証が要る
   ②`karume_ir` と同じく「配布物の中に処理系が生える」③検査が「実行してみるまで分からない」
   形になる。**数値変換は別定義の有限個の演算**（dtype・縮約軸・演算順・丸め・非有限の扱いを
   固定し、変換後テンソルの比較 fixture を Python / TS 双方に持つ）として切り出す。

## Consequences

- **破壊変更である**。配布形・manifest（`karume/4` → `karume/5`）・IR（v1 → v2）・公開 API の
  読み口が同時に動く。CHANGELOG の Breaking に載せる。未リリースではないので、これは
  「旧 pin は動き続ける・新 pin は新形式」という**版の分岐**として扱う（決定 18）。
- **退役するもの**: ADR 0063 の書き出し順規約・`repack.py` 不変条件②（`karume_ir` の逐語同一 —
  IR v2 で再 serialize するため。①生バイト同一は**継承する**）・piece キーの綴り規約
  （`<名前>#00002-of-00003`）・shard 0 のグラフ専用規約・末尾ゼロ詰めの読み手側分岐 3 つ・
  `weightChannelAxes`（決定 11）。
- **失うもの**: HF の safetensors プレビュー（ただし i4 / i2 席では既に失われている）・
  「汎用 safetensors ツールで開ける」性質・公開 10 リポの再アップロードと pin 10 本の差し替え・
  `.safetensors` 前提の運用台本（`hf-upload.zsh` / runbook / 門番）。
- **得るもの**: 入口が 1 つになる・PTQ 席と fine-tune からグラフが独立する（`krg` 共有）・
  block 目次と検証の土台ができる（Range を足せば部分取得）・PLE などの資産が物理配置ハック
  なしで一級市民になる・無改変 safetensors と LoRA と部品差し替えが**同じ束縛表**で組める。
- **`krg` が持ち出せる資産は重み非依存のものだけ**という制約は、運用上「`krg` を配れば何でも
  再現できる」わけではないことを意味する。QAT の活性 scale がグラフ側にある（決定 1）のは
  その一例で、`krg` の同一性は**内容ハッシュと依存条件**で判定する（名前や「同アーキ」では
  判定しない）。
- **段 0 の宿題は閉じた**: `pushErrorScope('validation')` の同期区間を block 単位に割る費用は
  実測でほぼ無料（決定 9 — push/pop 1.81 µs / 回）。費用の主はフェンス（13.0 ms / 回）なので、
  フェンスの粒度は part（または予算単位）に留める。
- 低 bit codec の**速度**は本 ADR の主張ではない（決定 15）。三値を入れてもデコードは速く
  ならない見込みで、効くのはメモリだけである。速度の主張をするなら ADR 0105（packed int8
  活性）側の実測が先に要る。

## 段階分解と検収

| 段    | 作るもの                                                                                                                                                                                                                                                                                                 | 検収                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0** | 本 ADR + [container-v1](../container-v1.md)。**CPU 試作 4 本**: ①container の読み書き（2 文書 descriptor・const 領域・単一 / 分割・`krg` のバイトコピー抽出）②anima transformer の上流重み → 束縛表（名前対応 1:1・bf16 → f16）③LoRA → 書き換え済みグラフ（`parseIrGraph` を通す）④既存 codec 4 種の台帳 | ①往復で**バイト同一**（書く → 読む → 書くで 1 バイトも動かない）・抜いた `krg` が独立生成の `krg` とバイト同一 ②DiT 567 + conditioner 118 本が 1 本残らず束縛表に載る（不足・余剰ともに 0） ③書き換え後グラフが `parseIrGraph` を通り、scale 0 で元グラフと構造一致 ④台帳の `decodeCpu` 4 本が現行 `decodeI8` / `decodeI4` / `decodeI2` と**全要素ビット一致**（三値は i2 経路の部分集合として） / GPU が空いたら errorScope 分割の費用を計測して本 ADR に追記                                                                                                                                            |
| **1** | IR v2（`storage` 外出し + scale の rank 2 統一 + `rowAxis` + **直列化規則**〈キー順・数値の綴り〉）・TS リーダ（descriptor + `BlockSource`・**part 単位取得**）・Python writer と**移行 CLI**・単一 / 分割・QAT 込み・`verify_lanes` への新レーン登録                                                    | ①**主 = CPU の逐語突合**: 旧 shard 列と新 `krm` から initializer ごとに実体 / scale のバイト列・宣言 shape・格納 codec・group を取り出し sha256 が**128 鎖全本一致** ②**77 グラフ全部で新旧の適格述語**（`plan.ts:451` / `:496` / `:519`）**の結果が一致** ③同一環境で新旧を走らせ**出力バイトを直接比較** ④既存の環境別 sha256 参照行（ADR 0106）と golden を**維持**する。**`KARUME_REFERENCE=write` / `rewrite` で緑にしない**（golden は許容差の回帰網であってビット同一の門ではない） ⑤**同じグラフを再 export するとグラフ記述がバイト同一**（`krg` の同一性を内容ハッシュで判定する条件 — 決定 4） |
| **2** | manifest `karume/5`・hub の **block 単位**取得 / キャッシュ / 検証・PLE の asset 化（専用 part）・extras の移行・部品差し替え席・1 系列の再アップロード                                                                                                                                                  | ①`fromPretrained` が新 pin で緑・越境参照が通る ②**RAM ピーク harness の新設**: 同じブラウザ版 / モデル / quant / 設定で **cold / warm / ローカル**を分け、Session 準備完了までの external 最大値と取得バッファ・展開 scratch・持越し scale を併記（複数回） ③warm で digest が **0 回**であることを計数で示す ④部品差し替えで、重みを 1 バイトも取る前に admission が不足 / 余剰を全件列挙して落とす                                                                                                                                                                                                     |
| **3** | 全資産 / 全 pin の移行・門番 / runbook / `hf-upload.zsh` / `karume dist` の追随・**part 長の既定の見直し**                                                                                                                                                                                               | ①全レーン緑・CHANGELOG Breaking ②`distribution_gate` / `assets_gate` が manifest の**中身**を見る ③part 長を 256 から動かした構成で RAM ピーク harness を再測し、宣言からの事前見積りと実測が一致する                                                                                                                                                                                                                                                                                                                                                                                                     |
| **4** | anima transformer f16 の上流取り込み（bf16 デコーダ・比較 fixture・`index.json`・大ファイルのストリーム取得）                                                                                                                                                                                            | 上流重み → TS 経路の出力テンソルが配布形のバイトと**1 本残らず一致**（f16 席）。RTN 経路は許容差                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **5** | LoRA 実行時 A/B（PEFT → kohya）                                                                                                                                                                                                                                                                          | ①adapter を読み込んで scale 0 = 元グラフと**完全一致** ②同じ復元済み base + 同じ A/B 式の CPU 参照と一致 ③未消費テンソル 1 本で例外 ④追加ノード数・失った融合・dispatch 数・実時間を記録（PEFT との品質差は別評価）                                                                                                                                                                                                                                                                                                                                                                                       |
| **6** | 新 bit 幅 / 三値カーネル・Range 取得・ロード時合成・再融合                                                                                                                                                                                                                                               | 各実測                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

**段 5 の見積り（CPU 試作 ③ の実測）**: anima transformer の実 IR は**ノード 2603 / `linear`
454 本 / `static_quantize` 0 本**である。

- 差し込みは対象 1 本につき **+3 ノード**（low / delta / add）で、**`mul` ノードは足さない**
  （alpha / r は B 側へ畳む）。454 本全部に差し込むと 2603 → 3965 ノードになる。
- **ゼロ bias の新設は 0 本で済む**。exporter は定数を内容ハッシュで名付けて重複除去しており
  （`tools/exporter/src/karume/convert.py:635-678`）、この命名規約を TS 側で再現すると既存の
  ゼロ bias 6 本が **6/6 再利用できた**（delta 枝の bias は長さ = 出力次元なので base の bias と
  必ず同名になる）。
- **454 本全部に差し込んでも `parseIrGraph` を通過する**。
- 差し込みで**外れる融合は `linearStaticQuantize` の 1 本だけ**
  （`packages/runtime/src/runtime/fusion-rules/linear-static-quantize.ts`）。anima は
  `static_quantize` が 0 本なので実質 0 件である。

## 追記 1 — 段 1 の実装で確定した点（2026-09-22）

段 0 の設計を実装に落とす過程で決めた / 訂正した点。仕様の正本は [container-v1](../container-v1.md) と
[ir-v2](../ir-v2.md) で、ここは「なぜそうしたか」だけを持つ。

1. **initializer 名 = 実体の鍵**（IR v2）。v1 の initializer 名は torch.export の placeholder 名で、
   上流の鍵（FQN）は `tensor` 欄が別に持っていた。`tensor` 欄を束縛表へ外出しすると、上流 checkpoint の
   取り込み（決定 19）と LoRA の対象解決（段 5）が鍵を失う。名前を FQN（定数は `const.<hash>`）にすれば
   表を 1 つも足さずに済む（ミラー 77 グラフで名前・キーは全て ASCII・最長 87 文字・衝突 0 を実測）。
   共有 initializer は**借り手の名前 = 貸し手の initializer 名**（`shared.tensor` は不要になった）。
2. **正準直列化の規則**（決定 4 の条件）: 空白なし・キーはスキーマ順・名前キーの map は code point 順・
   数値の綴りは ECMAScript `Number::toString`（JS 側は `JSON.stringify` がそのまま正準で、Python 側が
   これを実装する）。descriptor の配列順も固定（`const.blocks` は offset 昇順 MUST・`blocks` は
   (part, offset)・`constants` は (graph, initializer)）。
3. **仕様の訂正 5 点**（container-v1 §「改訂履歴」）: ①`capabilities.codecs` はモデル記述の `codecs` へ
   （グラフ記述は `krm` / `krg` でバイト同一 MUST なので、束縛表に依存する欄を持てない）②束縛表から
   借用形を外す（shared 宣言は IR 側だけ。束縛表のキー集合は「shared でも const 供給でもない
   initializer」と完全一致）③`rowAxis` / `groupSize` / `scale` は量子化 codec のみ ④`int8-sym` の packing
   は 1 要素 / 1 バイト / 整列 4（4 / 4 / 1 だと `numel % 4 == 0` という v1 に無い制約が入る）
   ⑤配列順の固定（上 2）。
4. **`rowAxis` は宣言が正本だが、消費側 op の軸との突合は残す**（§13.2 の「`weightChannelAxes` は消える」を
   訂正）。宣言だけで scale の意味は閉じるが、消費側と食い違う宣言は GPU 常駐経路が scale を別の軸に
   当てる沈黙誤値になるので、`planWeightResidency` が突合点として持つ。group 形（`int4-sym-g`）の
   `rowAxis` は 0 だけ（展開カーネルと `decodeI4` が先頭次元を行とする）。
5. **要素数 0 の退化形**（`in_features = 0` など）: per-channel の `groupSize` は行長 0 では 1 以上 MUST を
   満たせないので **1**、group 数は **1**（per-channel scale は行ごとに 1 本あり、旧配布形の `[rows, 1]`
   と一致）。§6.1 の `scaleShape` の式はこの退化形を含む。
6. **共有 initializer の門**: 借り手は格納を宣言しないので、旧 5 点のうち「宣言 格納 dtype が貸し手と
   一致」は消え、代わりに**貸し手の codec と借り手側の消費（適格判定）から期待席を導いて**貸し手の
   実際の席（i8 / i2 は行の軸まで）と突き合わせる。
7. **段 3 までの暫定接着**（両読みではない）: 旧配布形の読み手は残るが、①旧 v1 ローダ（`parseIrGraph`）は
   読んだ時点で合流後の形へ写す（改名・codec 写像・`rowAxis` / `groupSize` の導出）②旧 shard の validator
   は供給計画と同じ `ReadyInitializer`（バイト列 + rank 2 の scale）を返す。消費側（Session 構築・
   常駐計画・models）は**合流後の 1 語彙**だけを見る。旧形式を読むコードは段 3 で削除する。
8. **errorScope は block ごと・フェンスは part ごと**（決定 9 の実装）: 供給の単位 `WeightBatch`（旧 shard
   1 本 / コンテナの part 1 本）ごとに空 submit + 完了待ちを 1 回、errorScope はコンテナでは item
   （block）ごと、旧 shard では batch ごと（従来どおり）。
9. **`krg` 単独では Session を組めない**（重みの供給が無い）— 設計どおり。const 供給と shared だけの
   グラフは例外的に組める。
10. **公開面**: `openContainer` / `prepareContainer` / `createSessionFromContainer` / `codecLayout`（codec →
    展開経路。models が格納の性質で分岐する唯一の読み口）/ `ContainerFormatError`。
11. **検収③の縮図**を `gpu_container_session_test.ts` に固定: 同じ重みバイト列から `krm` 経路と旧 safetensors
    経路が同じ出力バイト列・同じ常駐バイト数・同じ見積りを出す（i4 + group scale / f16 / i8 + per-channel
    scale・piece 分割込み）。
12. **Python の書き手（段 1）は 1 コンテナ 1 グラフ**（`write_model_container(..., graph_name=…)`・
    bindings はテンソルキー直下の flat map）。現行配布形の部品（component）1 本 = グラフ 1 本に対応
    する。1 コンテナに複数グラフ（sbv2 の 4 グラフ級・決定 20 の「quant 席 1 つに container 1 つ」）を
    載せる口は段 2（manifest `karume/5`）の裁定で足す。資産（`assets`）の受け口も同じく段 3
    （PLE sidecar の専用 part）まで作らない。`provenance.writer` は呼び手が渡す（自動で焼くと版を
    上げるたびに言語横断 fixture のバイトが動く）。part 長は天井（1024 MiB）だけを強制し、集合
    `{256, 512, 768, 1024}` は既定値と定数で示す。
13. **`__proto__` キーは両側で塞ぐ**（container-v1 §0）。
14. **移行 CLI（段 1）の面**: コンポーネント単位（1 コンポーネント = 1 グラフ = 1 コンテナ）・manifest は
    読まない / 書かない・`--license` 必須・グラフ名の既定は親ディレクトリ名・`provenance.writer` の
    既定は生成器タグ・自己検査は payload 部で突合・`.partial` → 検査 → 据え替え（container-v1 §12）。
    `karume verify` のコンテナ席（移行済み資産を CLI から検査する口）は段 2 で足す。
15. **検収の状況（段 1 時点）**: ①CPU 逐語突合は実ミラー 3 コンポーネント（depth-anything-v2 small /
    depth f32・sbv2 shared / text_encoder i8 と i4）で initializer 1,281 本の sha256 が一致。128 鎖全本は、
    ディレクトリを跨ぐ shard 列（上 14 の未対応）を旧 manifest から引く経路が段 3 で入ってから閉じる。
    ②77 グラフ全部で新旧の適格述語 3 本・codec 写像・`rowAxis` / `groupSize` が改名表を通して一致
    （initializer 18,515 本・量子化 4,533 本・不一致 0）。③縮図（追記 11）は緑。④参照行は 1 行も書いて
    いない。⑤Python が書いた fixture を TS が開いて `parse → serialize` がバイト同一、writer の決定性は
    pytest で固定（同じグラフを 2 度書いてバイト同一）。

## 追記 2 — 段 2 の裁定（2026-09-22）

段 2（manifest `karume/5`・hub の取得面・PLE の asset 化・部品差し替え席・1 系列の再アップロード）の
計画で決めた点。manifest の形は ADR [0109](0109-manifest-v5-container.md) が正本で、ここは本 ADR の
決定に対する訂正と補足だけを持つ。

1. **決定 20 の訂正 — コンテナの粒度は quant 席ではなく部品 × dtype**。`quants[].container` は
   `weights.<部品>.<dtype>.container` に置き換わり、quant 席は `karume/4` と同じ写像のまま。根拠は
   実ミラー 11 本の集計 — 席ごとに物理ファイルを作ると 65.9 GiB が 197 GiB（×2.99）になる
   （quant 席は重みの単位ではない: gemma4 の 3 席は重みが同一で実行ノブだけ違う）。「1 コンテナに
   複数グラフ」は形式の能力として残り、`karume/5` では使わない。model 単位の `assets`（tokenizer
   等・越境あり）は manifest に残す。共有 `krg` の `graph: FileRef` 席は `karume/5` に置かない
   （要る段 = 段 5 で足す）。
2. **段 2 の取得単位は part**（決定 6 の記述どおり）。hub はコンテナ 1 本につき `BlockSource` を
   返す面を 1 本持ち、cold は取得層の相 1（ファイル全量を流しながら sha256 検証 + 記録ハッシュ）で
   part を温めてから区間読み口を開く。区間読みの費用型で分岐する — seek（ブラウザの Blob・ローカルの
   区間読み）は block ごと、scan（Deno の既定）は part を 1 度に読んで切る。取得層
   （`@hdae/fetch-cache`）は段 2 では変更しない。**HTTP Range は段 6 のまま**で、前倒しの条件は
   「2f の RAM ピーク harness で cold のピークが『part 長 + 重ね合わせ』を超える」こと。
3. **決定 8 の補足 — block の sha256 は未検証の取得元にだけ掛ける**。取得層がファイル全体を検証した
   バイト列（HF 経由）は `BlockSource` が検証済みと名乗り、`readBlock` は digest を掛けない
   （cold の 2 重 digest を避ける）。`fromContainer(bytes)` とローカルディレクトリ（ADR 0086 決定 2 —
   sha256 を照合しない取得元）は block ごとに digest する。warm は従来どおり 0 回。block の sha256 は
   段 6 の Range 取得で「届いた分だけ検証する」ための契約として残る（container-v1 §7 を訂正）。
4. **追記 1 の 12 の訂正 — assets の受け口は段 2 に入れる**（段階分解表が正本）。小段の最後に置き、
   PLE は asset（役割 `ple-values` / `ple-scales` の block 列 — 区間読みの block は 1 block = 1 part — と
   役割 `ple-index` の索引 schema 3）、`extras` の `rope_base` は asset（役割 `rope-base`・66 KB × 2 本の
   複製）へ移る。資産は `assets[].length`（payload 長）を宣言する（ADR 0109 決定 4）。
5. **段 2 で `karume/5` を書くのは移行 CLI のリポ丸ごとモード**（`karume migrate --manifest`）。
   旧 manifest から shard 列を引くので、追記 1 の 14 / 15 で未対応だったディレクトリを跨ぐ shard 列と
   128 鎖全本の逐語突合はここで閉じる。dist.py / recipe が `krm` を直接書くのは段 3
   （container-v1 §12 の記述どおり）。`hf-upload.zsh` の `*.krm` 追随だけは再アップロードに要るので
   段 2 へ前倒し。
6. **共存期間**: 2b 以降 hub は `karume/5` だけを読む。ローカルミラー 11 本は 2d で全部移行して
   差し替え、レーンは移行済みミラーで回す。HF の pin は段 2 で irodori-v4.1-small だけ更新し、残りは
   段 3 まで旧版パッケージからだけ動く。
7. **再アップロードする 1 系列 = irodori-v4.1-small**（5.8 GiB・部品 8 × dtype 4・quant 5 席・
   extras / PLE / 越境なし — 部品と席の写像を一番広く踏み、移行 CLI の制約に当たらない）。
8. **RAM ピーク harness（検収②）は Deno（`--expose-gc` + `Deno.memoryUsage().external`）で自動化**し、
   Chrome は手動確認手順として渡す。取得（part 0 / const / block）を別々に数える。
9. **小段の順**: 2a 仕様 → 2b hub → 2c runtime → 2d exporter（リポ丸ごとモード + 全ミラー移行 +
   128 鎖突合）→ 2e models（8 系列の container 経路・部品差し替え席・PLE / extras の asset 化）→
   2f 検収（harness・再アップロード・docs 同期）。

## 追記 3 — 段 2 の実装で確定した点（2026-09-22〜23）

manifest の形は ADR [0109](0109-manifest-v5-container.md)、PLE は ADR [0085](0085-ple-host-gather.md)
追記 2026-09-22 が正本。ここは段 2 の実装で決めた / 訂正した点と検収の状況だけ。

1. **資産は論理長を宣言する**（`assets[].length` — container-v1 §2.2 訂正 2 ⑤）。資産は shape を持たず
   descriptor から payload 長を復元できないので、消費側が末尾の 0x00 詰めを推測で剥ぐ形になっていた
   （PLE 索引の JSON がその実例）。block 長 = 論理長の 4 の倍数への切り上げ MUST を parse で突き合わせる。
2. **区間読みの資産は 1 block = 1 part、全量読みの資産は資産どうしで part を共有してよい**（重み block とは
   同居しない）。書き手の `AssetInput.dedicated_part` が区別する。gemma4 E2B の `model` 容器は 83 part
   （重み 8 + 資産 74 + descriptor + const）になる。
3. **hub の取得面は温めを持たない**（`openContainerSource` は同期・区間読みだけ）。温めは呼び手が
   `prefetchAssets` で行い、順序は descriptor（part 0）→ admission → 重みの part → 資産。取得面の中で全 part を
   温めると「実行できないモデルの重みは 1 バイトも落とさない」が面の内側から壊れる。併せて `runPrefetchPhase`
   の相 1 の能力判定を ref ごとにし、ローカルセッション + HF 越境の温めが飛ぶ既存不具合を直した。
4. **models の継ぎ目**（`hub/components.ts`）: `ModelComponent` は `graph` / `createSession` に加えて容器の
   資産の宣言（`assets: 名前 → 役割`）と読み口（`asset(name)`）を持つ。admission は各部品のグラフ宣言と
   資産の宣言で判定し、重み block と part を共有しない資産（索引・`rope_base`）はそこで読んでよい。
   全量面 `assetComponentOpener` は同期の供給口を返すために全部品を先に開く（admission より先）。
5. **部品差し替え席**は `ComponentSource = { source, model?, quant? }`（別の `karume/5` リポの同じ役割）。
   admission = ①グラフ記述の sha256 が manifest の宣言と一致 ②束縛の不足 / 余剰 0（`openContainer`）
   ③家族の門。①は 2 つの manifest だけで判定するので descriptor すら取らずに落ちる。QAT のように活性
   scale をグラフへ焼く系列では checkpoint ごとにグラフ記述が変わり、差し替えは拒否される（段 5 で
   「グラフ同一性の方針」を宣言するまでの規律）。
6. **移行 CLI のリポ丸ごとモードで実ミラーから拾った 3 点**（合成 manifest だけのテストでは見えなかった）:
   `pipeline` は `"<name>/<major>"` の文字列・`gemma4-qat` も PLE の持ち主・karume-gemma4（非 QAT）の PLE
   索引は schema 1（I8・`storage` 欄なし）。schema 3 では `storage` を必ず綴る（`i8` へ正規化）。
7. **検収①（CPU の逐語突合）は閉じた**: ミラー 11 本を `karume migrate --manifest` で移行（69 GiB・自己検査 =
   旧 shard の payload と initializer ごとに一致）し、TS の読み手（`openContainer` + `readBlock`）で
   一意な容器 108 本・block 40,939 本・initializer 31,882 本・66.8 GiB の sha256 と合流（不足 / 余剰 0）を
   全件通した（越境 4 容器は karume-anima 側と同一実体）。「128 鎖」はモデルごとに数えた延べ数で、
   共有部品を畳むと 108 本。
8. **段 2 の間の共存**: hub は `karume/5` だけを読む。ローカルミラーは `models/`（移行済み）と、旧
   `karume/4` のミラーはリポ外 `~/workspace/karume-models-v4/`（git 追跡外）に置き、レーンは `models/` で
   回す。HF の pin は irodori-v4.1-small だけ段 2 で更新した（検収①の実 pin: SHA 固定の取得元で
   `fromPretrained` → `generate` を通し、取得 26 本が全て pin の revision・同じ文と seed の WAV が
   ローカルミラー経由と byte 同一）。残りの pin は段 3 まで旧版パッケージからだけ動く。
9. **PLE 側で変わった規律**: 既定の常駐上限は「最大 block 2 本ぶん」（約 64 MiB — 旧は最大 shard 2 本 ≈
   506 MiB）。読み口は費用型を持たないので seek / scan の方針表は消え、全量読みへ倒す下限は
   `min(32, block の行数)`。読み 1 本を途中で畳む口は無い（`AssetReader.read` は signal を受けない）ので
   中断は gather の段の境目だけ。
10. **検収②③（RAM ピーク harness・warm の digest）**: `tools/ram-peak/matrix.ts` を新設し、実資産 3 構成
    （gemma4 e2b `i4-fast` / irodori v4.1-small `i8-a8` / anima turbo `f16+dit8-a8-attn8-s16`）で cold →
    warm → local を各 3 回測った（[研究記録](../research/2026-09-23-container-ram-peak.md)）。warm は
    3 構成とも **payload の digest 0 回・キャッシュ書込 0 本**（descriptor 2 文書の突合は開くたびに掛かる —
    container-v1 §7 の①）。ホスト RAM のピークは seek 型（ローカルの位置読み・ブラウザの Blob）で
    「part 1 本 + block 1 本」の見積りに収まり、Deno の HF 経由（scan 型）はその約 2 倍（part 2〜3 本が
    同時に生きる — 取得層の新規バッファ + hub の 1 枠 + runtime の part 単位の items）。改善候補は段 3
    （「part 長の既定の見直し」と同じ回）で実測して採否を付ける。Range 取得（段 6）の前倒し条件
    「cold のピークが part 長 + 重ね合わせを超える」には、seek 型では当たらず scan 型で当たる — ただし
    scan 型の超過は取得ではなく保持の重複が原因なので、Range ではなく上の候補で閉じる。

## 追記 4 — 段 3a〜3d の実装で確定した点（2026-09-23〜24）

書き手を `krm` の 1 本にし（3a）、系列出力を移し（3b）、テストと道具を追随させ（3c）、旧配布形の
読み手を削除した（3d）回で決めた / 訂正した点と、段 3 の検収①②の状況。part 長の既定と RAM ピーク
（検収③）は追記 5 が持つ。形式は [container-v1](../container-v1.md)、manifest は ADR
[0109](0109-manifest-v5-container.md) が正本。

1. **書き手は 1 本**（3a・`8086ed37`）。recipe の `publish_model` / `export_to_file` と `karume migrate` は、
   同じ公開の 3 段（書く → 読み直して検証 → 据え替え — `karume.publish.publish_container`）を通る。
   書き手が 2 本あると「移行済みミラー」と「再 export した系列」が別物になるため。`provenance` と
   `graph_name` は必須にした（既定値で出所を偽らない・作業席名 `<部品>.staging` をグラフ名に拾わない）。
   同一性は「旧 shard → `karume migrate`」と「直接書き」が part 列ごとバイト同一であることで固定した
   （f32 / f16 / i8 / i4・`rope_base`・PLE の専用 part）。`karume repack` と旧 shard の書き手は退役し、
   旧形式を読むのは移行専用の `karume.legacy` だけになった。追記 2 の 5 の「dist.py / recipe が `krm` を
   直接書くのは段 3」はここ（recipe 側は `d2630521`）で済んだ。
2. **`karume dist` は `karume/5` を組み、現物を宣言と突き合わせる**（3a・3b）。`weights.<部品>.<dtype>.container`
   （2 文書の期待値 + part の FileRef 列）を書き、容器のグラフ名の集合が weights のキーと違えば組み立ての
   前に落とす（`DistError`）。IR の受理規則（ランタイム支援 + op 契約 — `assert_runtime_support` /
   `assert_op_contracts`）は export / `karume dist` / `karume verify` の 3 経路で同じ関数を掛ける。据えた容器の
   2 文書から IR を起こし直して掛ける（`verify.ir_graph_from_container`）のは dist と verify の 2 経路で、export は
   書く前のグラフ（`stored.graph`）に直接掛ける。
3. **追記 1 の 14 の訂正 — 容器のグラフ名 = 部品名 = `karume.json` の weights のキー** MUST（3b・`2bbd013b` /
   `0b86439e`）。規則の本文は container-v1 §2.1、なぜ規則にしたかは ADR 0109 追記 1 が持つ。ディレクトリ名は
   規則にしない — 系列直下に容器を置く family ではディレクトリ名が系列名になり、irodori の `caption-proj`
   （キーは `caption_proj`）や deberta の `full-24layer`（キーは `text_encoder`）のように綴りも違う。recipe は
   グラフ名を定数で名乗り、AST の門 `tools/export-recipes/tests/test_graph_names.py` が「定数であること」と
   「名乗るグラフ名が配布計画の weights のキーと対応すること」を見る（配布形に載らないグラフを名乗る
   family・他 family の計画と突き合わせる family といった例外の扱いもこの門が持つ）。現物の側は上の 2 の
   `karume dist` の門が、容器のグラフ名と weights のキーを検査する。
4. **追記 1 の 14 の訂正 — `provenance.writer` は既定で書かない**（3b・`2bbd013b`）。既定で生成器タグを
   焼くと、移行した容器と recipe が直接書いた容器の part 0 が版の分だけ永久に食い違い、同じ資産の
   バイト同一が主張できない。ツールの版は `karume.json` の `generator` 欄 1 箇所が持つ。
5. **全資産の移行**（3b・3c）。`outputs/series/` の旧 shard 形 54 系列 / 135 部品 / 70.9 GiB を
   `python -m migrate_series` で `krm` へ移した（値はビット同一・sidecar は容器の資産・golden と json は
   不変 — `0b86439e`）。「系列 → 部品 → weights キー → sidecar の畳み方」の対応は family しか知らないので、
   core に系列モードは作らず、recipe 側の駆動に置いた（ADR 0065 の境界）。PLE（gemma4 / gemma4-qat）と
   `rope_base`（anima）は recipe が容器の資産として直接書く（`d2630521`）。git 追跡の golden fixture
   32 モデルは `karume.goldens` で焼き直し（旧 safetensors 50 本 → part 78 本・`io.*` は不変 —
   `2bde1baf`）、packed PLE fixture は recipe の本番経路で再生成した（`f5736ac0`）。実証は 2 組ある。
   ①`karume dist` の焼き直し（siglip2 / gemma4 / gemma4-qat）は、現ミラーと重み・資産の part でバイト同一。
   ②recipe の直接 export（vowel-detector / deberta 3 variant / gemma4-e2b-product）は、移行結果と全 part で
   バイト同一。
6. **追記 1 の 7（段 3 までの暫定接着）を閉じた**（3d・`9e905d9f`・決定 18）。runtime から、旧 safetensors
   方言の読み手・shard の進行検証・IR v1 ローダが消え、Session の供給元は容器 1 種（`BoundContainer`）に
   なった。`parseSafetensors` は素の safetensors の付帯資産を読むために残し、方言 dtype `I4` / `I2` だけ
   拒否する。hub は shard の逐次面（`streamAssets`）と使い回しの器（`into` / `readFileInto`）を削除した。
   テストと道具は 3c で容器面へ移した（実資産 e2e は part 列の位置読みで開く
   `packages/runtime/tests/helpers/container-files.ts`、系列 → 部品 → グラフ名の表は門番と e2e が共有する
   `helpers/series-graphs.ts` の 1 本）。
7. **決定 17 の合流層は `packages/runtime/src/format/container/bind.ts`**（3d）。2 文書に依存しない
   `bindDeclarations` と、2 文書から引き当て口を組む外皮 `bindGraphs` の 2 段で、`krm` もメモリ内容器（下の 8）も
   同じ `bindDeclarations` を通る。旧 `parseIrGraph` の storage 規則は 2 箇所に分かれた。scale /
   `groupSize` / `rowAxis` の有無（量子化 codec では必須・それ以外では禁止）は記述文書の読み手（`descriptor.ts` の `parseEncoding`）が見て、メモリ内容器は
   入力形が違うので同じ有無を `memory.ts` で見直す。値の側（codec × 意味論 dtype・`groupSize` の値域・行長の
   整除・i2 経路の shape・scale 長）と、piece の被覆・中間 piece の長さ 4 の倍数・整数の行境界は `bind.ts` が
   持つ。IR の読み手は格納を持たない IR v2 の宣言 1 種（`parseIrDeclaration` /
   `parseIrDeclarationValue`）である。本文・段階分解表・段 5 の見積りに残る `parseIrGraph` / `prepareModel`
   と、対象欄の `format/container.ts` は段 0 時点の名前で、書き換えない。
8. **メモリ内容器**（3d・`9e905d9f`）— `openMemoryContainer`（`packages/runtime/src/format/container/memory.ts`）は、
   手元のバイト列を**容器ファイルを書かずに** `krm` と同じ合流・admission・構築の経路へ載せる供給面で、
   `openContainer` と同じ `BoundContainer` を返す。対象は models がホストで組むグラフ（irodori の CFG 合成 /
   Euler 更新・gemma の最大値選択・PLE gather）と、道具の合成モデルである。旧 `createSession(gpu, model)` が
   持っていた「手元のグラフを Session にする」口を、形式を増やさずに容器面へ寄せるために置いた。
   - **入力**: `graphs`（グラフ名 → IR v2 宣言）と `tensors`（グラフ名 → initializer 名 → 供給）。shared で
     ない initializer は過不足なく供給する MUST（不足も余剰も全件列挙で落ちる）。
   - **供給の 2 形**: `bytes`（丸ごと 1 本 — 渡した器をそのまま返し、複製しない）と `pieces`（先頭次元の
     行範囲 + `read()` の読み口を 2 本以上 — バイト列を抱えず、`readBlock` のたびに `read()` を引き直す）の
     排他。量子化 codec は `encoding` に f32 の scale（rank 2・4 B 整列）と `groupSize` を持つ。
   - **part の割り方**: 丸ごとの供給は宣言順に積み、既定の part 長（256 MiB）を超えるところで次の part へ
     移る（companion scale は実体と同じ part）。piece は 1 本 = 1 part で、scale は piece 1 と同じ part
     （規則③）。part は Session 構築のフェンスの単位なので、この割り方が決めるのは構築時の staging の
     上限である（ホスト RAM は part の割り方に依らない — 追記 5 の 3）。
   - **sha256 は掛けない**。渡されたバイト列はネットワークもディスクも通っていないので、宣言と現物の
     食い違いを検証する相手が無い（block の digest は未検証の取得元のための門 — 決定 8）。
   - **検査の門**: 束縛規則は上の 7 の合流層が `krm` と同じ 1 本で見る。この層が持つのは、上の 7 の
     encoding の有無の見直しと、合成に閉じた 5 つである — 宣言との対応（不足 / 余剰・未宣言のグラフ名）、
     `pieces` が 2 本以上、合成した block id の衝突、scale のバイト位置の 4 B 整列、`readBlock` の取得長と合成した宣言長の一致。ホストで組む宣言は
     必ず公開面の `parseIrDeclarationValue`（`openContainer` が容器の宣言に掛けるのと同じ門）を通す。
     `krm` 経路では記述文書の読み手が非有限数と入れ子の深さを検査するが、メモリ内容器ではこの 2 つを
     呼び手が持つ（`packages/runtime/mod.ts` の doc）。
9. **追記 1 の 10 の公開面の増減**（3d・公開面スナップショット）。runtime は −8 / +9 で、削除は `openModel` /
   `KarumeModel` / `ModelShard` / `createSession` / `createSessionFromShards` / `prepareModel` /
   `estimateSessionMemory` / `ContainerError`、追加は `openMemoryContainer` / `BoundContainer` /
   `MemoryContainerInput` / `MemoryEncoding` / `MemoryPiece` / `MemoryTensor` / `parseIrDeclarationValue` /
   `IrDeclaration` / `RuntimeSupportError`。hub は −3（`streamAssets` / `StreamedAsset` / `StreamAssetsOptions`）、
   models は増減なし。追記 1 の 10 の 5 つは残る。capability 不足は `RuntimeSupportError`、容器の規則違反は
   `ContainerFormatError` に分かれた。見積りは `prepareContainer(...).estimate()` が持つ。
10. **追記 1 の 11 の縮図の後継と、独立オラクルの喪失**（3d）。`gpu_container_session_test.ts`（`krm` 経路と
    旧 safetensors 経路の A/B）は旧経路と一緒に削除した。後継の `packages/runtime/tests/gpu_memory_container_test.ts`
    は、同じ合成モデル（linear i4 + group scale → add f16 → linear i8 + per-channel scale + bias f32・piece
    分割込み）で `krm` とメモリ内容器を突き合わせ、piece の割り方は 2 経路でわざと違えてある。ただし
    2 経路は合流層から Session 構築までを共有するので、この縮図の出力を**別実装で**押さえる A/B は
    無くなった。合流か構築の誤りは両辺に同じだけ乗り、この 1 本では検出できない。見積りの A/B
    （`runtime_prepare_model_test.ts`）も同じ形である。緩和として外部の正解が 2 つ残る。①codec ごとの
    GPU テストは CPU 参照（`applyReferenceOp` + `decodeI4` / `decodeI8`）と突き合わせる。②実資産の環境別
    sha256 参照行（ADR 0106）は段 3 の間 1 行も書き換えていない（最終変更 `689157c7`・2026-09-20）ので、
    行がある環境では新しい経路が旧経路で焼いた出力とビット同一であることを押さえている（anima /
    irodori / sbv2）。3 codec 混在 × piece 分割の縮図には外部の正解が無く、これを戻す作業は
    [backlog](../backlog.md) の later に置く。
11. **検収②は成立**（3c・`dcd6fb1c`）。`packages/runtime/tests/distribution_gate_test.ts` は `karume.json` を
    hub のパーサで読み、`karume/5` を名指しで断言し、既定の model / quant が選ぶ全容器の part の実在と長さを
    宣言と突き合わせる（越境参照の part はローカルミラーに実体が無いので対象外）。`assets_gate_test.ts` は manifest を持たない系列について、容器を part 列として開き
    （本数・長さ・2 文書の parse・束縛表との合流まで）、名乗るグラフ名が期待の weights キーであることを
    見る。どちらも block の sha256 は読まない（実バイトの突合は各 e2e が `openContainer` の経路で通る —
    決定 8）。射程の限りとして、配布形の門番が見るのは既定 quant だけで、他の quant 席の part は e2e が
    開くまで検査されない。
12. **検収①は条件つき成立**。CHANGELOG の Breaking は `994bdad8` で記載した。レーンはこの開発機（Intel
    Arc B570・Deno 2.9.6）で 14 本を回した。3d のコミット本文（`9e905d9f`）は単独再走込みで全 14 レーン緑と
    記録している。3e 後の通常走行では 12 / 14 が緑で、赤の 2 本は [known-issues](../known-issues.md)
    「Intel Arc B570」節の環境要因である。①`test:core` の OOM 門
    （`gpu_generation_context_test.ts`「state 確保の失敗は out-of-memory errorScope で fail loudly」— `destroy()`
    の解放が次の device poll まで遅れる）。②`test:models:birefnet` の 2048² が device lost になり、Deno の
    panic で走行ごと止まる（1024² の 11 本は緑）。ADR 0005 のリリース判定（緑必須）は、この 2 本を緑に
    できる環境で満たす。
13. **全 pin の移行は段 3 から release の波へ移す**（段階分解表の段 3「全 pin の移行」と、追記 2 の 6・
    追記 3 の 8 の「残りの pin は段 3 まで」の訂正）。段 3 で pin は 1 本も動いておらず、`karume/5` を指すのは
    段 2 で上げ直した irodori-v4.1-small だけである。残り 9 リポの再アップロードと pin の差し替えは
    リリースの回に行う。[release-runbook](../release-runbook.md) §0 の順序（version bump → 焼き直し →
    アップロード → pin → JSR publish）が非可換 MUST で、manifest の `generator` 欄がパッケージ版を写すので、
    bump 前に上げると古い版を名乗る配布形が HF に載るため。再アップロードでは新 manifest が参照しない
    旧ファイルを同じリポから消し、pin は削除後の main で焼く（irodori-v4.1-small で実施 — 旧 safetensors
    65 本・5.84 GiB を削除して pin を付け替えた `1c952948`。旧 revision の pin は履歴から解決できる）。
    リポの削除 → 再作成は、公開版の pin を持つリポでは使わない（決定 18 の「履歴は残す」）。台本
    `hf-upload.zsh` は段 2 で `*.krm` に追随済み（`f33f7be2`）。manifest 側の訂正は ADR 0109 追記 1。
14. **block 上限の見直し（32 → 16 MiB）は段 3 では測らなかった**。決定 5 は「段 3 の RAM ピーク実測まで
    持ち越す」としていたが、段 3e で測ったのは part 長だけである（追記 5）。見直しは段 6（新 bit 幅・
    Range 取得の回）へ持ち越す。下げる前に要るものは決定 5 のとおり（const の上限を超える定数の逃げ道
    〈container-v1 §10 の未決〉と、block 件数・descriptor の倍増の評価）。

## 追記 5 — 段 3e の実測で確定した点（2026-09-24）

段 3 の「part 長の既定の見直し」と、段 2 から持ち越した RAM ピークの改善候補 3 つの採否。実測の正本は
[研究記録](../research/2026-09-24-part-length-ram-peak.md)（3 系列 × part 長 {256, 512, 1024} MiB ×
cold / warm / local × 3 回の中央値）で、ここは決めた点と検収③の状況だけ。

1. **part 長の既定は 256 MiB のまま**（決定 6 の「段 3 の検収後に見直す」への答え）。下の 2 / 3 の後、
   seek 型（ローカルの位置読み・ブラウザの Blob）のピークは part 長に依らなくなったが、Deno の HF 経由
   （scan 型）は hub の保持枠が part を握るので、おおむね part 長とともに伸びる（cold の anima run 1,139 / 1,741 /
   2,245 MiB。gemma4 load は 1,038 / 1,032 / 1,515 MiB で、256 → 512 では伸びていない）。
   既定を上げて得るのは part 本数だけで、区間読みの資産は 1 block = 1 part の専用 part なので本数もほとんど
   減らない（gemma4 E2B の `model` 容器は 83 / 79 / 78 part）。
2. **候補 3 を採る — hub の scan 型の切り出しは保持枠の器の view**（`slice()` → `subarray()`）。写しが
   part ぶん 2 重に乗る形が消える。成立条件は整列で、器は tight（byteOffset 0）かつ block 開始は 64 B 整列
   （決定 7）なので、scale の `Float32Array` view に要る 4 B 整列がそのまま満たされる。区間より長く持つ
   呼び手は写す（`AssetReader.read` の MUST）。
3. **候補 2(c) を採る — Session 構築は block を 1 本読んでは上げて手放す**（`WeightBatch.items` の
   lazy 化・フェンスは part ごと 1 回のまま）。根拠は WebGPU 仕様の `writeBuffer`（呼び出しの時点で写す —
   決定 9 の追記・ADR 0070 決定 3 の追記）で、`packages/runtime/tests/gpu_write_buffer_copy_test.ts` は
   Arc B570 / Deno 2.9.6 の `test:core` レーンで緑。
4. **候補 1（取得層の `readFile` に器を渡して使い回す）は採らない**。単独の削減は候補 3 以下で、候補 3 と
   組むと器の view が hub の外へ出て、runtime 公開型 `BlockSource` の寿命契約を借用に変える破壊的変更が要る
   （3 と衝突する）。加えて、取得層は同じ器の並行使用を拒むので PLE の行読み（16 本並行）が直列になり、
   器が最大 part 長で常駐して gemma4 の host PLE の decode 中に約 +224 MiB 増える見込み（**推測**）。
5. **検収③（part 長を動かした構成の再測・宣言からの事前見積りとの一致）**: seek 型は成立 — 3 と 2(c) の
   後は part 長に依らず external 最大 71〜139 MiB で、item 1 本（最大 block 32 MiB + 同乗 scale + 展開席の
   f32 展開結果）と固定分の見積りの桁に収まる。展開席の piece 列の間は、これに持ち越し scale の写し
   （scale 全量）が 1 本加わる（式の正本は container-v1 §11）。scan 型は「保持枠 1 本 + GC を待つ前の器」で、GC を待つ
   本数は宣言からは閉じない（part 256 の anima で最大 part の約 4.4 本 = run 1,139 MiB）。warm の payload
   digest 0 回・キャッシュ書込 0 本（段 2 の検収③）は全 27 構成で保たれた。
6. **Range 取得（段 6）の前倒し条件**（追記 2 の 2）について、追記 3 の 10 の見立て（scan 型の超過は保持の
   重複が原因で、Range ではなく段 3 の候補で閉じる）は半分だけ当たった。重複は 2 / 3 で消えたが、scan 型の
   cold は今も「part 長 + 重ね合わせ」を超える（part 256 で irodori 582・gemma4 1,038・anima 1,139 MiB）。
   残りの原因は part 単位の全量読みそのもの（取得の粒度）で、条件は形式上成り立ったまま。前倒しするかは
   ここでは決めない。
