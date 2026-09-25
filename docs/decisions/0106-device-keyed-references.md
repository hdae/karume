# 0106: sha256 参照値を環境キーごとの行で持つ

- Status: accepted（2026-09-20 — テスト整理の波 段 1。実装は HEAD に入っている）
- Date: 2026-09-20
- 対象: `packages/runtime/tests/helpers/environment.ts`（環境キー）/ `helpers/reference.ts`（参照値
  fixture の読み書きと参照門）/ `helpers/results.ts`（結果の席）・単体テスト
  `packages/runtime/tests/environment_key_test.ts` / `reference_fixture_test.ts` /
  `results_writer_test.ts`・追跡 fixture `packages/models/tests/fixtures/references/{anima,sbv2,irodori}.json`・
  消費側の e2e 5 本（`e2e_anima_test.ts` / `e2e_sbv2_wav_test.ts` / `e2e_irodori_wav_test.ts` は
  参照値 + 結果、`e2e_birefnet_real_test.ts` / `e2e_depth_anything_real_test.ts` は結果のみ）。
  2026-09-22 追記の対象は `packages/runtime/tests/e2e_golden_test.ts`（golden 判定の 2 段目 =
  WGSL 仕様帯を環境キー別に + 実測の記録）・実重み golden 11 本の e2e（`<系列>-golden` の席）・
  `tools/verify-diff/`（複数環境の `results.json` を並べる読み口）。
  **ランタイム・配布形・exporter は無改変**（テストと検証道具の持ち物だけ）
- 関連: ADR [0005](0005-verification.md)（検証戦略 — 「全ケース SKIP は明示 FAIL」の規律と
  2026-09-20 追記のレーン分割）/ [limitations](../limitations.md) の「sha256 参照門」節
  （クロスデバイスのビット同一を保証しない機序）/ [known-issues](../known-issues.md) の
  「Intel Arc B570」節（載せ替えの実測）/ [assets-layout](../assets-layout.md)（`outputs/` の根）

## Context

e2e の移植の門（`e2e_anima_test` / `e2e_sbv2_wav_test` / `e2e_irodori_wav_test`）は、出力 PNG /
WAV の sha256 が期待値と**ビット一致**するかだけを見る。この門が主張できるのは「**その機で**数値が
1 ビットも動いていない」ことで、クロスデバイスのビット同一は仕様として保証しない（機序は
limitations — 超越関数のドライバ依存・シェーダコンパイラの fma 融合判断・Tint / naga の経路差）。

期待値はテストソースの定数 1 本だった。そのため、2026-09-20 に開発機の GPU を RTX 3080 Ti から
**Intel Arc B570**（BMG G21 / Mesa ANV / Linux xe）へ載せ替えてフル verify を回したところ、
**sha256 参照門 16 本が全て赤**になった（出力 PNG は目視で正常 = 数値の微小差・known-issues）。
このとき門は「退行を掴む」役目を失っている — 全部赤なので、本物の退行が 1 本混ざっても見分けが
付かない。かといって tolerance 化は禁止で（緩めると「移植できた」の意味が消える）、定数を新しい機で
焼き直せば今度は旧機の退行検出器が消える。**1 つの席を 2 台で奪い合っている**のが問題の形だった。

併せて、割れたときに見る**実物**（PNG / WAV）の置き場も無かった。一致した回の実物が手元に無いので、
次に割れたときの A/B の材料が割れた後にしか作れない。

## Decision

### 1. 参照値は**環境キーごとの行**で持つ（追跡 JSON）

置き場は `packages/models/tests/fixtures/references/<系列>.json`（git 追跡下）。形は
`{ "schema": 1, "kind": "sha256", "cases": { "<ケース ID>": { "<環境キー>": "<sha256>" } } }` で、
読めない・JSON として壊れている・`schema` / `kind` が違うは全て throw する（空として扱わない）。

書き出しはケース名・環境キーとも**辞書順**・2 スペース・末尾改行で安定させる。複数環境の行が 1
ファイルに同居する席では、差分が「実際に変わった行」だけになることが必須条件になる。

現況（2026-09-25 時点）の行数は anima 11 ケース / sbv2 6 ケース / irodori 2 ケース × 環境 2 本
（`deno-intel-graphics-bmg-g21` と `deno-nvidia-geforce-rtx-3080-ti`）。anima の extra 2 ケースは
`deno-intel-graphics-bmg-g21` の行だけを持つ。

### 2. 環境キー = `<ランタイム>-<アダプタ名 slug>`

`helpers/environment.ts` が `GPUAdapterInfo` から作る。基底は `description` が空でなければそれ、
空なら `${vendor}-${architecture}`（Deno は `description` に GPU 名を入れ `architecture` が空、
Chrome は `vendor` / `architecture` に名前を入れ `description` が空 — どちらかは必ず埋まる）。
小文字化 → 商標の飾り `(r)` / `(tm)` / `(c)` を落とす → `[^a-z0-9]+` を `-` へ → 前後の `-` を除去。

- Deno + `Intel(R) Graphics (BMG G21)` → `deno-intel-graphics-bmg-g21`
- Chrome + vendor `apple` / architecture `metal-3` → `chrome-apple-metal-3`

名前が 1 つも採れない環境では **throw する**（空キーの行へ黙って混ぜると、別の機の参照値を「この機の
参照値」として突き合わせることになる）。GPU アダプタが取れない環境は環境キーを**持たない**
（空文字などで埋めない）。綴りの規則そのものは `environment_key_test.ts` が固定する。

**ドライバ版はキーに入れない。** 理由は 2 つで、① WebGPU API からドライバ版は採れない ②入れると
ドライバ更新のたびに「新しい行が無い」= 明示 SKIP へ黙って逃げてしまう。入れなければ、ドライバ更新で
数値が動いたときに**同じキーの行が割れる** = 赤で気づく形になる。

**OS もキーに入れない**（2026-09-21 裁定）。同じ GPU を複数の OS で使い回す機は現状無いので、
キーを太らせると既存の行の綴りが変わるだけで、防げる衝突が実在しない。衝突が実際に起きたら
（同じアダプタ名の別 OS が同じ席を奪い合う形）そのときに再考する — 結果 JSON（決定 5 の
`results.json`）は `environment` に OS を持つので、どの機の行かは後から辿れる。

### 3. 作るモードは 3 つ（環境変数 `KARUME_REFERENCE`）

| モード    | 行が無いケース                             | 行があるケース                            |
| --------- | ------------------------------------------ | ----------------------------------------- |
| 未設定    | 明示 SKIP（登録時に作り方を 1 度警告する） | 比較（不一致は赤・tolerance 化は禁止）    |
| `write`   | 実測を**追加**して緑                       | 比較（**上書きしない**）                  |
| `rewrite` | 実測を追加して緑                           | 実測で**上書き**して緑（旧→新を印字する） |

未知の綴りは throw する（黙って「未設定」に落とさない）。`rewrite` は「何が変わったのかを先に
言えるとき」だけの操作で、**他環境の行には決して触らない**。3 モードの意味論は
`reference_fixture_test.ts` が実 GPU 無しで固定している（環境とモードを引数で注入する）。

### 4. 参照門（`registerReferenceGate`）と opt-out

「この環境の参照値がまだ 1 件も無いので sha 門が全て SKIP された」を**無音の緑にしない**ための
門番を、各 sha ファイルの末尾に 1 本置く。ADR 0005 の「全ケース SKIP は明示 FAIL」と同じ規律で、
opt-out も同型の環境変数 `KARUME_ALLOW_NO_REFERENCE=1`。GPU も資産も無くて sha 門自体が走らない
環境ではこの門番も鳴らさない（`runnable` で渡す）。失敗文面は行の置き場と
`KARUME_REFERENCE=write` の作り方を出す。

**数えるのは、そのファイルが今回登録したケース ID 集合だけ**である（`registerReferenceGate` は
`warnMissing` と同じ集合を受け取る）。

- 登録したケースの**どれにも**現環境の行が無ければ赤。
- 登録したケースの**一部**にだけ行がある状態は緑（ADR の字義どおり — この門が言うのは
  「この環境の参照値が 1 件も無いのではない」ことだけで、全ケース検証済みとは言わない）。
- ケースの改名・削除で残った**孤児行**（もう誰も突き合わせない行）は数えない。fixture 全体を
  横断して 1 行でもあれば緑にすると、孤児行 1 本が現役ケース全 SKIP を緑で隠す。
- 作るモード（`KARUME_REFERENCE`）が立っているときは、行が 0 件でも緑（その走行が行を作るため）。
  リリース判定機にこのモードを残さないこと（[release-runbook](../release-runbook.md) §1）。

### 5. 結果の席は `outputs/verify/<環境キー>/<日付>_<系列>/`

参照値（追跡が要る・環境ごとの行）と結果（追跡外・走らせるたびに増える）は性格が違うので席を分ける。
結果は **消して安全**な側で、`rm -rf outputs/verify` で常に作り直せる（[assets-layout](../assets-layout.md)
の `outputs/` 根の流儀・綴りは exporter 側 `tools/export-recipes/_shared/paths.py` の `VERIFY_ROOT`
と対）。置くのは 2 種類だけ:

- `results.json` — `schema` / `family` / `environment`（ランタイム名・版・V8・TypeScript 版 +
  アダプタ 4 欄 + OS）/ `checkout`（HEAD の sha と dirty 真偽。`git` が無ければ `null`）/
  `startedAt` / `cases[]`（`id` / `status` = `pass` \| `fail` \| `written` \| `rewritten` /
  `expected?` / `actual?` / `artifact?` / `elapsedMs` / `note?`）。1 件積むたびに丸ごと書き直すので、
  途中で落ちても直前までが残る。
- **実物**（PNG / WAV）— **成功・失敗を問わず毎回**。不一致のときだけ残す形だと、一致したときの実物が
  手元に無く、次に割れたときの A/B が採れない。

同じ日・同じ系列を 2 度走らせたら**最後の走行が残る**（日付までで席を分け、走行ごとには分けない —
分けると「最新はどれか」を人が数えることになる）。置き場の決定も作成も**使うときまで遅らせる**
（GPU 無しの機でもモジュールは読まれるので、開いた時点で環境キーを要求すると「SKIP されるはずの
ファイル」がモジュール評価で落ちる）。

### 6. 双子ケースは「双子の**行**と実測」で突き合わせる

別経路から同じバイトが出るはずのケース（anima の `fromPretrained-512` / `onEvent-1024` /
`base-cfg-fromAssets-shards`）は、**自分のケース ID の行**を持ちつつ、双子のケースの行と実測が
一致することも検査する。参照値が環境ごとの行になってからは、経路間のビット同一を「同じ定数を
両方へ渡す」形では持てないためで、相手の行がこの環境にまだ無いときだけ検査が成立しない。

### 7. RTX 3080 Ti の行は記録上の adapter 名から導いた綴り（実測未確認）

既存の定数はそのまま `deno-nvidia-geforce-rtx-3080-ti` の行へ移した。この綴りは記録に残る adapter
`description`（`NVIDIA GeForce RTX 3080 Ti`）へ決定 2 の規則を当てて**導いた**もので、その機で
実際に名乗らせて確かめてはいない（GPU は既に載せ替え済み）。実機の綴りが違っていた場合は「その環境の
行が無い」= 明示 SKIP + 参照門の赤になるので、**誤った行と黙って突き合わせることはない**
（`environment_key_test.ts` にこの綴りの導出が 1 件入っている）。

## 検討した代替案

1. **テストソース内の参照値の配列に「環境」列を足す**（定数表を 2 次元にする）— 却下。行の追加が
   毎回コード変更になり、`KARUME_REFERENCE=write` のような「実測が自分の席を作る」形にならない。
   参照値は機械が書いて機械が読むデータで、レビューで読むコードではない。環境が 1 台増えるたびに
   e2e 3 本を編集することになり、レビュー差分がハッシュの羅列で埋まる。
2. **参照値を `outputs/` の下に置く**（結果と同居させる）— 却下。`outputs/` は 3 根とも git 追跡外
   （[assets-layout](../assets-layout.md)）なので、clone 直後に突き合わせる相手が無く退行検出器に
   ならない。参照値は「前回この機で出たバイト」ではなく「**この機が出すべきバイト**」の宣言なので、
   追跡される側に置くのが正しい。結果（追跡外・毎回増える）との分割が決定 5。
3. **不一致を tolerance で吸収する**（sha を画素差の閾値に替える）— 却下。ADR 0005 の規律どおりで、
   緩めた時点で「移植できた」の意味が消える。デバイスが違うときの健全性検証は参照 sha との一致では
   なく**自己 A/B**（同一入力・幾何 2 種または新旧 2 版の出力 sha の一致）で行う（limitations）。
4. **golden の許容差も追跡 fixture へ出し、実測から行を起こす**（決定 1 と同じ形を 2 段目に当てる）—
   却下。帯が走行のたびに実測へ追随する形になり、ADR 0005 の tolerance 化禁止と衝突する。帯は
   「この GPU の実装はここまで外れてよい」という**仕様由来の宣言**で、実測から起こす値ではない。
5. **結果を追跡下の台帳に集め、環境間の差異を門にする**（`tools/verify-diff` を赤で落とす）— 却下。
   クロスデバイスのビット同一は非保証なので、差異そのものを赤にできない。追跡外の結果を追跡下へ
   持ち込むことにもなり、参照値と結果の席の分割（決定 5）が崩れる。

## Consequences

- **クロスデバイスのビット同一は依然として非保証**。行が増えても保証が増えるわけではなく、増えるのは
  「その機での退行を検出できる台数」だけ。limitations の by-design はそのまま残る。
- **新しい機での最初の走行は全ケース明示 SKIP + 参照門の赤**になる。`KARUME_REFERENCE=write` で同じ
  レーンを回すと行ができるが、これは**その回の実測を正とする**操作なので、その機の健全性は別途
  （出力の目視・自己 A/B）で確かめてから行を作る。
- fixture に複数環境の行が同居するので、レビューでは「どの環境の行が動いたか」を見る（辞書順固定 +
  `rewrite` が現環境の行しか触らないことで、それが差分から読める）。
- 系列が増えるたびに fixture 1 本と参照門 1 本が増える。`results.json` は環境の素性とチェックアウトを
  持つので、複数デバイスの結果を後から突き合わせる材料になる（突き合わせる道具は追記決定 3）。

## 追記（2026-09-22）— golden 側の環境別化と結果の突き合わせ

### 追記決定 1: golden の 2 段目（WGSL 仕様帯）も環境キーごとの行で持つ

`packages/runtime/tests/e2e_golden_test.ts` の `OUTPUT_TOLERANCE` は `<model>/<出力名>` →
**環境キー** → `{ spec }` の 2 段の表にする。2 段目が発火するのは**走らせている機の行がある出力
だけ**で、行が無い機では 2 段目そのものが無い（1 段目の Karume 独自基準だけで測り、超えれば赤）。
1 段目・fail の文言・`note` の記録は不変。

行を分けるのは、**緩めを足した機の外へ緩めを広げない**ため。`activations/sin` の atol 2⁻¹¹ は
Intel Arc B570（Mesa ANV）の実測で足した行で、キーが `<model>/<出力名>` だけだと、同じ op を
ほぼ正しく丸める機（RTX 3080 Ti は 1e-6 で通る）の退行検出の網まで同じだけ緩む。

決定 1 と並べる意味は違う。golden の期待値は **torch CPU 由来で device 非依存**なので、期待値を
機ごとに持つ必要はない。機ごとに並ぶのは**許容差**の側である（sha256 の参照値は期待値そのものを
機ごとに持つ — 決定 1）。

### 追記決定 2: 合格した回の差も `results.json` に残す（任意欄 `measurements`）

`cases[]` の 1 件に任意欄 `measurements` を足す。1 本は `output`（グラフの出力名）/ `maxAbs` /
`maxRel` / `tolerance`（受理に使った帯）/ `stage`（`karume` \| `spec` — fail のときは最後に測った段）。
許容差の判定は落ちたときにしか数値を見せないので、合格した回の差はどこにも残らない。毎回残せば、
帯を緩めるかどうかの判断材料が割れる前から手元に揃う。

- **派生値は持たない**（帯に対する比などは読む側 = `tools/verify-diff` が導く）。同じ数の別表現が
  2 か所に乗ると、片方だけ直った形が作れてしまう。
- `schema` は **1 のまま**（任意欄の追加で、既存の読み手は無影響）。非有限（NaN / ±Inf）は
  `JSON.stringify` が `null` にするので、読む側は `null` を受ける。
- 判定・帯・assert・`ignore` はこの欄と無関係（記録であって判定材料ではない）。

実重み golden 11 本（sbv2 / irodori / birefnet / siglip2 / deberta / depth-anything / dacvae /
embeddinggemma / minicpm5 / gemma4 / vowel-detector）にも結果の席を配線し、置き場は
`<系列>-golden` にする。models 側の e2e が素の系列名で同じ根へ書くので、席を分けないと同じ日の
同じ席を 2 つのファイルが奪い合う。

### 追記決定 3: 環境間の突き合わせは門ではなく読み取り専用の道具（`tools/verify-diff`）

読むのは `outputs/verify/<環境キー>/<日付>_<系列>/results.json`（決定 5 の席）だけで、何も書かない。
環境ごとに最新の日付の席 1 本を採り（`--date` でその日付に固定）、系列ごとの「ケース × 環境キー」の
行列と、`status` / 実物の sha / 片方にしか無いケースの差異、`checkout` の不一致・dirty の警告を
Markdown（`--json` 可）で出す。

**門にしない**。クロスデバイスのビット同一はそもそも非保証（limitations）なので、環境間の差異
そのものを赤にできない。終了コードは差異があっても 0 で、1 で落ちるのは読めない・`schema` が違う
ときだけ（黙って「差異なし」は出さない）。

2 台目の結果は**手で** `outputs/verify/<環境キー>/` へコピーする前提で、同期機構は置かない
（追跡外の結果を追跡下へ持ち込まないための席の分割 = 決定 5 と整合させるため）。別の置き場に
まとめてあるなら `--root` で指す。

## 追記（2026-09-25）— 結果の席の日付は UTC

決定 5 の席 `outputs/verify/<環境キー>/<日付>_<系列>/` の `<日付>` は **UTC** の日付である（走行を始めた時刻の
`toISOString()` の日付部 — `packages/runtime/tests/helpers/results.ts`）。挙動は変えない。

- ローカル日付とはずれる。JST の機では 09:00 より前の走行が前日の席に入る。「同じ日・同じ系列を 2 度走らせたら
  最後の走行が残る」の「日」も UTC の日である。
- `tools/verify-diff` の `--date` も UTC の日付で指す。ローカル日付で指すと別の席を引く。
- 「最新の日付の席」（追記決定 3）は取り違えない。UTC の日付は走行時刻に対して単調なので、最新の席は常に最新の
  走行を含む。
- ローカル日付にしない理由は、機の時刻帯で席の意味が変わるから。複数環境の結果を 1 か所に並べる読み口
  （追記決定 3）では、席の日付が全機で同じ基準である方が突き合わせやすい。
