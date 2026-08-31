# 0086: 取得元抽象 — `DistributionSource` と手元の配布形

- Status: accepted（2026-08-31 — 公開面レビューの裁定「ローカルローダー = b) DistributionSource
  抽象」+ 第 2 ラウンドの設計裁定〈骨格・確認 3 点・未 mapping の降格〉。実装は段①〜③で完了。
  裁定の原文 = `.claude/reviews/2026-08-31_182ced7/SUMMARY.md`）
- Date: 2026-08-31
- 対象: `packages/hub/`（`src/source.ts` = 内部契約 / `src/sources/{hf,local}.ts` = アダプター /
  `deno.ts` = サブパス / `src/session.ts` / `src/errors.ts`）・`packages/models/`（8 家族の
  `fromPretrained` と `src/hub/repo-ref.ts`）。**IR 仕様・ランタイム・配布形（`karume.json`）・
  exporter は無改変**。
- 関連: ADR [0038](0038-manifest-v1.md) §5（取得層の接続契約 — 本 ADR が「HF アダプターの契約」へ
  降格させる）/ [0080](0080-hub-fetch-cache-050.md) 決定 1（検証責務の移譲 — 検証は取得元ごと）/
  [0070](0070-shard-loading-admission.md) 決定 2（2 相ロード — 相 1 は optional 能力になる）/
  [0073](0073-models-source-pin.md)（`ref` 必須の MUST を取得元ハンドルへ継承）/
  [0008](0008-public-api.md)（公開面は薄い面のみ）

## Context

配布形は既に「ディレクトリ 1 つ」として手元に揃う（`models/karume-gemma4-e2b` = 3.8GiB のミラー・
`docs/assets-layout.md`）のに、**それを読む口が公開面に無かった**。公開面レビュー（2026-08-31）で
全レッグが一致した最大の blocker で、現状の唯一解は次の形だった:

- `examples/shared/local-dist-server.ts`（**非公開・196 行の疑似 HF サーバ**）を立て、実ポートへ
  HTTP で取りに行く。偽の repo 名と偽の commit SHA を名乗らせ、CacheStorage（Deno では `DENO_DIR`）
  へ **3.8GiB をもう 1 部複製**してから読む。初見の消費者が経路に辿り着くのに 6 ファイルを読む。
- `fromPretrained("./models/karume-gemma4-e2b")` は**綴りとしては HF の `owner/name` に見える**ため
  取得層まで滑り、返るのは 401 / 404 —「取得先が存在しない」という原因の遠い診断になる
  （`models/karume-gemma4-e2b` は完全に合法な repo 名なので、綴り門では救えない）。

一方 hub の取得経路は HF 固有の語彙（`repo` / commit SHA / `hubUrl` / CacheStorage / `fetch`）で
全面が組まれており、「取得元が HF ではない」という選択肢そのものが型に無かった。

## Decision

### 1. 骨格は opaque ハンドル + 内部 driver 契約（公開面はハンドル 1 つだけ）

公開面に出るのは**中身を持たないハンドル** `DistributionSource`（`src/source.ts`）と、それを作る
factory（`localDirectory` / `@karume/hub/deno` の `denoDirectory`）だけである。取得の実装は
内部契約 `SourceDriver` / `PinnedSource` に閉じ、`mod.ts` は `DistributionSource` を**型としてしか**
輸出しない。

内部契約は、現行の取得経路を畳んで残った **5 つの質問**そのものである:

1. 可変 ref → 不変な世代識別子（`SourceDriver.resolveGeneration` — **セッション唯一の解決点**）
2. `karume.json` 1 本の全量バイト（`PinnedSource.readManifest`）
3. ある `FileRef` の全量バイト（`PinnedSource.readFile` — sha256 / size の期待値つき）
4. ある `FileRef` を「RAM に載せずに、後で 3 が安く済む状態にする」（`PinnedSource.prefetchFile` =
   ADR 0070 決定 2 の相 1。**optional 能力**）
5. 越境 (repo, revision) → 別の取得元（`PinnedSource.originFor` — ADR 0038 §7）

MUST: 進捗・並行度（in-flight バイト予算）・中断の透過・tight view 検査・エラー文脈の組み立ては
**共通層の作法**として `src/fetch.ts` / `src/context.ts` に残す。取得元へ降ろすと、取得元が増える
たびに同じ不変条件を書き直すことになる。

MUST: **ハンドルに公開メンバを生やさない**。判別は**同一性**（`instanceof` を包んだ型述語
`isDistributionSource`）で行う — ブランド欄は利用者が偽造でき、構造判別（`"repo" in value`）に
すると `HubRepoRef` の綴り間違いが黙って取得元として通る。models 側の union（`ref | source`）を
捌く分岐も、この述語 1 本に集約する（`src/hub/repo-ref.ts` の `toManifestSource`）。

MUST: 面ごとの作法（`fetch` / `caches` / `headers` / `onCacheError`）は取得元の**生成時ではなく
`pin` の呼び出しごと**に渡す。取得元は「どこから取るか」だけを持ち、「どんな作法で取るか」は
面のオプションから来る — こうしないと `loadManifest` に渡した `fetch` が以後の `fetchAssets` にも
黙って効き続ける。

アダプター層はさらに 1 段細い: ローカル取得元の実体読みは `DirectoryAdapter`（`readFile(path,
{signal})` の 1 メソッド）へ委ね、`localDirectory` 自身は**純 Web 標準**（特定ランタイムの API を
1 つも参照しない）に保つ。将来席（OPFS / IndexedDB / File System Access の picker）はこの 1 面に
乗る（下の「将来席」）。

### 2. ローカル取得元の検証は size 厳密一致のみ（sha256 は記録を信頼する）

`localDirectory` の `readFile` は `FileRef.size` との**厳密一致**だけを門にし、**sha256 は照合
しない**。ADR 0080 決定 1 と同型の哲学である — あちらは「格納時に検証し、以後は記録を信頼する」
で全量再ハッシュを消した。こちらは「手元のファイルは配布元から取得した物ではなく**利用者の
資産**」で、毎起動の全量ハッシュ（数 GiB）に見合う脅威が無い。

size は読み終えた時点でタダで分かるので門として残す（途中で切れたコピー・別 quant の取り違えは
ここで落ちる）。**改竄検出はしない**ことを `docs/limitations.md` に by-design として記載する。

manifest 側の門も HF とは性格が違う: HF の `MAX_MANIFEST_BYTES` は「受信を途中で止める」防波堤
だが、ローカルは手元の実体に対する形式検査なので**読み切ってから**落として構わない。また
`parse` の throw はそのまま外へ出す（ローカルには evict すべきキャッシュが無いので、壊れた
manifest は毎回同じ `ManifestFormatError` で落ちるのが正しい）。

### 3. 越境は明示 mapping → 明示 fallback → fail loudly（暗黙の推測・暗黙の降格は禁止）

越境参照（`FileRef` の `repo` + `revision` — ADR 0038 §7）の解決順は 3 段で、**どこにも推測を
挟まない**:

1. `LocalDirectoryOptions.crossRepo` の**明示 mapping**（キー = manifest が宣言する `"owner/name"`・
   値 = その repo をまるごと提供する `DistributionSource`）。
2. 無ければ `LocalDirectoryOptions.fallback` の**明示した委譲先**（例 `fallback: hfHub()` 相当の
   リモート取得元）。宣言された (repo, revision) の座標へ寄せてから開く。
3. どちらも無ければ **fail loudly**（`crossRepo` に何を書けばよいかを message が示す）。

MUST NOT: **隣接する同名ディレクトリの推測**（「`../karume-anima` があればそれ」）。取り違えた
バイト列を黙って読ませる形で、決定 2 のとおり size が合えば通ってしまう。

MUST NOT: **暗黙のリモート降格**。未 mapping を黙って HF へ落とすと、オフライン前提で組んだ配布が
network へ出る。降格は 2 の明示 opt-in だけである。

なお `originFor` は「同じ取得元の別座標」を返す口であって**能力を落とす口ではない** — 越境先だけが
相 1 を持たない形は取得元契約の破れとして共通層が fail loudly する。

### 4. `@karume/hub/deno` サブパスは横断不変条件への opt-in carve-out

Deno のファイルシステムを読むアダプター（`Deno.readFile`）は**このサブパスだけ**に置く。
`CLAUDE.md` の横断不変条件「ランタイム依存は Web 標準 API のみ」に対する**明示的な carve-out**で、
条件は 3 つ:

- **本体（`mod.ts`）を通る限り `Deno` は 1 度も現れない** — carve-out はサブパス 1 本に閉じる。
- **ブラウザ向けのコードから import しない**（ブラウザは決定 1 の `DirectoryAdapter` を自分で
  渡す）。JSR の `exports` で `.` と `./deno` を分けてあるので、依存の向きは配布形で固定される。
- サブパスに置けるのは「ローカルの実体をどう読むか」だけで、**取得元としての意味論**（世代・
  検証・越境・進捗）は本体の `localDirectory` が持つ。

同型の carve-out を将来別ランタイム（Node / Bun）へ増やす場合も、この 3 条件を満たす限り本 ADR の
射程内とする（`@karume/hub/<runtime>` を足すだけで、本体と models は 1 行も変わらない）。

### 5. ローカル取得元は CacheStorage を通らない（バイト複製ゼロ・`prefetchAssets` は no-op）

ローカルは「温める」対象を持たないので**相 1（`prefetchFile`）を実装しない**。共通層は
`prefetchFile === undefined` を見て何もせずに抜け、逐次面は相 2（直接逐次読み）だけで同じ
RAM ピーク目標 O(最大 shard) を満たす（ADR 0070 決定 2 の読み替え = 下の追記）。

これは省略ではなく**ローカル取得元の最大の利点**である: 手元の 3.8GiB を CacheStorage へ写す形は、
ディスクを 2 倍使ったうえで読みが 1 段増えるだけで、得るものが 1 つも無い。`headers` / `fetch` /
`caches` の 3 ノブも**取得元ハンドルを渡した呼び出しでは 1 つも効かない**（HTTP 取得元専用の
語彙が公開面に出ているもの — 8 家族の `*FromPretrainedOptions` の doc に明記した）。

### 6. エラーと識別欄の語彙を一般化する（持たない身元を名乗らせない）

HF 固有の語彙が診断の**必須欄**だったのを、取得元ごとに持てるものだけを名乗る形へ変える:

- `SourceOrigin` を新設し、取得元が持つのは「自分は何者か」だけにする。必須は 1 行の名乗り
  `label`（HF: `repo owner/name @ <commit SHA>` / ローカル: `ディレクトリ <ラベル>`）と
  `integrity` の 2 欄で、`repo` / `revisionSha` は**持たない取得元は省く**。
- `IntegritySource` に `"local"` を足す（`"cache" | "network" | "local"`）。ローカルは取り直しても
  同じバイト列が返るので、`"network"` と違って**再試行が回復手段にならない**ことを型で分ける。
- `IntegrityError` / `HubFetchError` の `repo` / `revisionSha` を optional 化する。
- `LoadedManifest` を**クラス**にし、`repo` / `revisionSha` を optional・`hubUrl` を**削除**する。
  取得元そのものを内部欄（`#session`）として運び、資産 3 面（`fetchAssets` / `prefetchAssets` /
  `streamAssets`）は識別欄から取得元を**組み立て直さない** — 組み立て直す形は「HF なら repo と
  SHA から復元できる」という HF 固有の性質に全面が寄りかかり、復元手段の無い取得元を入れられない。

MUST: **持っていない身元を合成しない**。ローカル取得元に repo / commit SHA を名乗らせると、実在
しないリポを指す診断（= HF へ探しに行けという案内）を生む。疑似 HF サーバ経路が偽の repo / SHA を
名乗っていたのは、まさにこの穴を回避策で埋めていた形である。

## 将来席（本波では実装しない）

- **OPFS / IndexedDB / File System Access の picker**: いずれも決定 1 の `DirectoryAdapter`
  （`readFile` 1 メソッド）として同じ面に乗る。`localDirectory` は純 Web 標準なので、hub 側の
  変更は 0 行で足りる見込み（ブラウザ実装が入った時点で実測する）。
- **キャッシュ先を Cache API 以外へ差し替える**（OPFS 常駐等）: これは取得元の話ではなく**取得層
  （`@hdae/fetch-cache`）側の将来課題**である。HF 取得元がキャッシュを所有する構造は本 ADR で
  変わっておらず、差し替えるなら fetch-cache の面が先に動く必要がある。
- `pruneHubCache`（repo 単位の細粒度掃除）は ADR 0080 決定 5 のままキャッシュ保守波に残る。

## 検討した代替案

- **a) `fromDirectory(path)`（Deno 専用の入口を models に足す）**: 追加面が 1 関数で最小だが、
  ①`@karume/models` 全体が Deno API に依存する（決定 4 の carve-out が本体へ漏れる）②ブラウザの
  OPFS / picker が乗る席が無く、実需が出た時点で別の面をもう 1 本足すことになる。却下。
- **c) 疑似 HF サーバを公開面に内包する**（`examples/shared/local-dist-server.ts` を製品化）:
  既存経路をそのまま使えるのが利点だが、偽の repo / SHA と 3.8GiB のキャッシュ複製と実ポートの
  占有という**回避策の性質を製品面へ固定する**。決定 5 / 6 が消そうとしているものそのもの。却下。
- **ハンドルにブランド欄を生やす**（`{ __karumeSource: true }`）: 判別が構造で書けて hub 外にも
  綴れるのが利点。却下 — 利用者が偽造でき、`HubRepoRef` の綴り間違いを取得元として通す穴になる
  （決定 1 の MUST）。
- **ローカルでも sha256 を照合する**: 改竄検出が付くのが利点だが、3.8GiB の全量ハッシュを毎起動
  払うことになり、ADR 0080 決定 1 が network 経路から消したコストを**ローカルで買い戻す**。
  却下（決定 2・必要な運用は HF 経路を使う — limitations に記載）。

## Consequences

- **hub の公開面が 3 つ増える**（`localDirectory` / `DirectoryAdapter` + `LocalDirectoryOptions` /
  `isDistributionSource` + 型 `DistributionSource`）+ サブパス `@karume/hub/deno` の
  `denoDirectory` 1 つ。既存面（`loadManifest` / `resolveFiles` / `fetchAssets` / `streamAssets` /
  `prefetchAssets` / `clearHubCache`）は**名前も種別も不変**で、第 1 引数だけが
  `HubRepoRef | DistributionSource` の union に広がる。
- **breaking（未リリース面）**: `LoadedManifest` の `repo` / `revisionSha` が optional 化・
  `hubUrl` 欄の削除・`AssetPhase` 以外の識別欄に依存していたコードは追随が要る。
- 段分割と門（実装順）: ①hub 内部再編（HF を 1 アダプターへ畳む・**既存テスト無改変で緑**が門）→
  ②`localDirectory` + `denoDirectory` + fixture テスト（`packages/hub/tests/local_test.ts` /
  `deno_directory_test.ts`）→ ③8 家族の入口 union 化 + 実ミラー直読 e2e
  （`packages/models/tests/e2e_gemma4_directory_test.ts` — `denoDirectory(models/karume-gemma4-e2b)`
  → chat 温度 0 が既存 golden と**逐語一致**し、CacheStorage / network 不通過を敵対注入と呼び出し
  回数 0 で証明）。疑似 HF サーバ e2e は **HTTP 疎通門として 1 本残す**。
- 消費者の動線が「ディレクトリを指すだけ」になる: `denoDirectory("./models/karume-gemma4-e2b")` を
  `fromPretrained` へ渡す 1 行で、network も CacheStorage も通らずに実配布形が読める。必要な権限は
  root 以下への `--allow-read` だけ。
- ADR 0038 §5 の記述のうち HF に固有なもの（revision 解決・キャッシュ名前空間・`hubUrl`）は
  **HF アダプターの契約**として読む（§5 への追記が指し先）。取得元非依存の作法（進捗・中断・
  同時取得数・tight view・エラーの形）は共通層の契約として §5 のまま生きる。
