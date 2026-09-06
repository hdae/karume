# 0094: hub のキャッシュ在庫照会と選択単位の削除（参照勘定）— ADR 0080 決定 5 の消化

- Status: accepted（2026-09-05 — ユーザー裁定: 参照勘定は「同じ manifest の全在庫ありの他の選択が
  守る」・manifest を跨ぐ勘定は持たず、足りない分は次のロードで再取得する形で一旦実装）
- Date: 2026-09-05
- 関連: ADR [0080](0080-hub-fetch-cache-050.md)（決定 5 — repo 単位の細粒度掃除
  `pruneHubCache` をキャッシュ保守波へ先送り。**本 ADR がその波**）/
  [0086](0086-distribution-source.md)（取得元契約の optional 能力 — 本 ADR が ⑥⑦ を足す）/
  [0038](0038-manifest-v1.md) §7（越境参照 — 実体の持ち主は参照先 repo）/
  取得層 `@hdae/fetch-cache` の ADR 0010（429 / 503 の再試行）・0011（HF 層の受信上限）

## Context

anima-web がモデルマネージャー（ダウンロードとロードの分離）を実装し、「model × quant が
ダウンロード済みか」を判定するために取得層 `@hdae/fetch-cache` へ直依存して `listKeys` の内容キー
`["hf", kind, repo, path, sha256]` を manifest の `FileRef` と突合していた。公開 API と文書化された
キー式だけを使ってはいるが、**取得層のキー設計がアプリ側へ漏れている**（越境参照の repo 解決や
`fileRefKey` の一意化まで、hub が既に持つ規則をアプリが再現している）。あわせて「モデル単位で
キャッシュを消したい」要望があり、こちらは quant / モデル間でファイルを共有する（quant を替えても
tokenizer や f16 の text encoder は同じ 1 本）ため参照勘定が要る。

hub の掃除 API は名前空間まるごとの `clearHubCache` だけで、細粒度は ADR 0080 決定 5 が
「キャッシュ保守波の `pruneHubCache`」へ先送りしていた。

同じ要望の 3 件目「shard 分割で要求数が増え HF の 429 が出やすくなった — `Retry-After` に従って
再接続してほしい」は取得層の話なので、hub ではなく fetch-cache 側で実装した（その ADR 0010）。
その際に hub の受信バイトの門（`transport.ts`）のうち移植する価値のある部分（宣言を超えた時点での
打ち切り）も fetch-cache の HF 層へ寄せた（その ADR 0011）。

## Decision

### 1. 公開面は 2 本 — 単位は取得と同じ「model / quant の 1 組」

```ts
listCachedAssets(loaded: LoadedManifest, selection?: ResolveOptions, options?: { caches? })
  → { cached: FileRef[]; missing: FileRef[] }
evictCachedAssets(loaded: LoadedManifest, selection?: ResolveOptions, options?: { caches? })
  → { evicted: FileRef[]; kept: { ref; reason: "shared" | "cross-repo"; sharedWith: string[] }[] }
```

第 1 引数を `LoadedManifest` にするのは、repo と取得元（HF かローカルか）を持つのがこの値だけ
だから（`session.ts`）。素の `Manifest` を受けると repo を別引数で渡させることになり、越境参照の
規則を呼び手が再現する今の形へ戻る。参照列は `resolveFiles` の値を `fileRefKey` で一意化したもの
（`fetchAssets` と同じ規則・同じ順）。boolean の便宜関数は足さない（`missing.length === 0` で足りる）。

### 2. 在庫と削除は取得元の optional 能力（⑥ `inventory` / ⑦ `evict`）

キャッシュキーの綴りは取得層の所有物で、hub の共通層が組み立てると取得層の版が上がるたびに
「消したつもりで残る」形が生まれる。そこで `PinnedSource` に optional 能力を 2 つ足し、共通層は
参照を origin ごと（`crossRefOf` の (repo, revision) の組・無ければセッションの取得元）に分けて
問い合わせるだけにする。

- **HF 取得元**: `listKeys(["hf", "model", repo])` を repo ごとに **1 回**引いて (path, sha256) で
  突合する（参照ごとに引くと参照の本数だけキャッシュ全体の列挙が走る）。削除は 5 要素の完全キーで
  `evict`（プレフィックス意味論だが完全キーなので対象はそのエントリ 1 件）。キー式を綴るのは
  `sources/hf.ts` の 1 か所だけ。
- **ローカル取得元**: `inventory` は「渡された全部がある」と答える（相 1 を持たないのと同じ
  理屈 — 直接読める取得元では「後の読みが安く済む状態」が最初から満たされている。実体の欠損は
  読む時に落ちる）。`evict` は持たない — ディレクトリの中身は取得物ではなく利用者の資産で、hub が
  消してよいものが 1 つも無い。共通層は `HubError` で断る。

### 3. 参照勘定は manifest 1 本の中で、全在庫の他の選択だけが守る

- 同じ manifest の他の (model, quant) のうち**全参照が在庫にあるもの**が使うファイルは残す
  （`kept: "shared"`・`sharedWith` に `"<model>/<quant>"`）。アプリが「ダウンロード済み」と表示する
  選択と一致させる。
- **部分在庫の選択は守らない**。どのみち次に使うとき残りを取りに行くので、守らせると「消せないのに
  使えないファイル」だけが残る。
- **越境参照は参照元からは消さない**（`kept: "cross-repo"`）。実体の持ち主は参照先 repo で、参照元の
  都合で他人のリポの在庫を消すことになる。消したいときは参照先 repo の manifest を開いて消す。
- **manifest を跨ぐ勘定はしない**。例えば anima-extra は anima の text stack を越境参照しているので、
  anima 側の選択を消すと extra は部分在庫に戻り、次のロードで足りない分だけ再取得になる
  （キャッシュは正しさの要件ではなく最適化 — 壊れはしない）。より完全な形（複数 manifest を渡す・
  守る選択を明示する）は必要になったときに足す。
- もともと在庫に無い参照は結果に載らない。manifest 本体（`karume.json`）のエントリは対象外
  （URL キー・小さい — 丸ごと消すのは `clearHubCache`）。

### 4. 429 / 503 の再試行と受信上限は取得層側（hub は 0.7.0 に追随済み — 2026-09-06）

fetch-cache 本体に入れた（既定で有効・`Retry-After` 優先・無ければ 1 / 2 / 4 / 8 / 16 秒・最大
5 回・`onRetry` 通知・`signal` で中断・`retry: false` で従来どおり）。hub 側の追随は fetch-cache 0.7.0 の
公開（2026-09-05）後に行った:

- 依存を `^0.7.0` へ。これだけで HF の rate limit は取得層が既定で取り直す。
- `LoadManifestOptions.onRetry` を revision 解決・`karume.json`・資産（相 1 / 相 2）・越境先の全取得へ
  透過する。通知の型は hub 所有の `RetryDiagnostic`（`CacheDiagnostic` と同じく取得層の型を再輸出せず
  構造で一致させる）。再試行の方針は取得層の既定のままで、`retry` は hub の公開面に出していない。
- `transport.ts`（hub 自前の `fetch` ラッパ = content-length の事前突合 + 受信超過の打ち切り）を撤去。
  受信超過の打ち切りは取得層 HF 層の `expectedBytes` 上限へ移った。content-length の事前突合は
  移植しない（汎用ライブラリでは Content-Encoding 越しの誤検知になる）。これに伴い ADR 0038 §2 の
  「`content-length` が `size` と食い違った時点で abort」と「manifest 本体は取得中に 1MiB 超過で
  abort」は本決定が上書きする — `karume.json` の 1 MiB 上限は取得層に厳密一致なしの上限が無いため
  **全量受信後の判定**（`parseManifest`）になる（limitations に記載）。
- HF 取得元の受信超過は `IntegrityError` ではなく `HubFetchError`（`cause` = 取得層のエラー）になる
  （sha256 不一致と同じ形）。`IntegrityError` を投げるのはローカル取得元だけになり、`SizeViolation` から
  発見場所（content-length / body）の引数を落とした。

## 検討した代替案

- **anima-web の現行形（fetch-cache 直依存）を認める**: 公開 API だけを使ってはいるが、越境参照の
  repo 解決と一意化をアプリが再現している。hub が同じ規則を内側で持てば直依存を返上できる。
- **`isSelectionCached` の boolean 版**: API 面が増えるだけ（`missing.length === 0`）。
- **参照勘定を「1 ファイルでも在庫がある選択が守る」/「全選択が守る」**: 前者は削除で空く容量が
  減り、後者は共有ファイルが事実上消えない（「共有しなくなったら消せる」の要望を満たさない）。
- **429 の再試行を hub の transport ラッパに置く**: hub に継ぎ目はあるが、他の下流（yomi /
  sbv2-web）に効かず、取得層が既に持つ「HTTP エラーの扱い」と二重になる（ユーザー裁定で
  fetch-cache 側）。

## Consequences

- hub の公開 API は追加のみ（次の minor）。取得元契約は内部（`mod.ts` は輸出しない）。
- `CacheStorage` が無い環境では、キャッシュを持つ取得元（HF）が全て `missing`・削除は空（取得層の
  契約どおり。hub で特別扱いしない）。ローカル取得元は決定 2 のとおりキャッシュを介さないので、
  `CacheStorage` の有無に関わらず「全て在庫あり」と答える。`Cache.keys()` 未実装のランタイム
  （Deno 2.8 以前）では取得層が fail loud に throw する。
- anima-web は `@hdae/fetch-cache` への直依存を返上できる。
- 将来席: manifest を跨ぐ参照勘定（複数 manifest・守る選択の明示）/ 参照先 repo だけを対象にする
  削除面 / 「取得元の能力不足」を専用エラー型へ切り出す（今は基底 `HubError`）/ ローカル取得元の
  在庫を実在検査にする（照会のたびにディレクトリを舐める I/O を払うなら）。

## 追記（2026-09-06 — 同一参照集合の兄弟席と `protect` / `alsoEvicted`）

anima-web の実機で「どの席を消しても evicted が 0 件」が出た。配布 manifest の anima は `f16` /
`f16-c16` と `f16+dit8` 系 4 席が**同一の参照集合**（重みは同じで session 設定だけ違う）を指し、
決定 3 の「対象以外の全組が守る」では兄弟席が互いを守ってどの順でも解けない。裁定（2026-09-06）:

- **既定の守る側から、対象と参照集合が同一の選択を外す**。同一集合の席はキャッシュの粒度では
  区別できず「片方だけ消す」は定義上できないので、守る側に数えると 1 本も消えないのに「守った」と
  名乗る嘘になる。真部分集合・上位集合の選択は従来どおり全在庫なら守る。
- **`CacheInventoryOptions.protect?: readonly ResolveOptions[]`**（Consequences の将来席「守る選択を
  明示する」）: 指定時はこの一覧だけが守る側の候補で、同一集合の除外はしない — 兄弟を名指しで
  守れば従来どおり `kept: "shared"`。守り方の方針（例: 厳選席 × カタログ全モデル − 対象）はアプリが
  持つ。対象自身は無視、存在しない model / quant は `ManifestReferenceError`。`listCachedAssets` には
  効かない。
- **`EvictedAssets.alsoEvicted: readonly string[]`**: 巻き添えで部分在庫に落ちた選択のラベル
  （呼び出し前に全在庫で、実際に消えた参照を 1 本以上使っていたもの — `evicted` と同じく取得元が
  消したと名乗った集合で判定）。アプリは「落とし済み」表示をこの一覧ぶん取り下げる。
- 影響: 同一集合の席を持つ manifest でだけ既定の結果が変わる（0.11.0 では 0 件だった削除が通る）。
  公開 API は欄の追加のみ（次の minor）。
