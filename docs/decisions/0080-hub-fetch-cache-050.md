# 0080: hub の資産キャッシュを fetch-cache 0.5.0 の内容キーと記録ハッシュ信頼へ移行

- Status: accepted（2026-08-28 — ユーザー裁定: 検証移譲 OK・認証隔離は撤去〈gated 運用予定なし・
  過剰防御〉）。**旧番 0080 ドラフト（`archive/hub-cas-0.5.0` の 272c0e9 — main 未マージ）を
  置換する** — 当時の裁定（検証責務の移譲・knob なし常時トラスト・self-heal の位置づけ・prune の
  先送り）は本 ADR がそのまま引き継ぎ、キーの実装形だけが変わった（hub 自前の合成 URL キー →
  fetch-cache 0.5.0 の内容キー）。
- Date: 2026-08-28
- 関連: ADR [0038](0038-manifest-v1.md)（fetch-cache 接続契約 — 決定 5 を本 ADR が更新）/
  [0070](0070-shard-loading-admission.md)（決定 2 の「sha256 照合はキャッシュヒットにも走る
  （非交渉条件）」を本 ADR が**変更** — 照合の実体が hub の全量再ハッシュから fetch-cache の
  記録ハッシュ比較へ移る）

## Context

fetch-cache 0.5.0 が公開された（breaking — 上流 ADR 0006〜0008）。要点は「**バイトは格納時に
検証し、ハッシュをエントリへ記録し、以後のヒットは文字列比較で決める**」への転換で、multi-GB
資産の起動ごと全量再ハッシュが消える。HF 内容キー `["hf", kind, repo, path, sha256]` は
revision に依存せず、revision bump でバイト不変なら再 DL しない — 旧 0080 ドラフトが hub 自前の
合成 URL キー（`https://karume.invalid/sha256/<hex>`）で作ろうとした性質を、ライブラリが
natively 提供する。一方 0.5.0 は `cacheName` を全廃した（名前空間 `"fetch-cache"` 固定）ため、
hub の認証別名前空間（`karume/1:auth:<hex16>` — gated 資産の token 間隔離）は席を失った。

## Decision

### 1. 検証責務を fetch-cache へ移譲する（knob なし常時トラスト）

資産取得（`fetchAssets` / `streamAssets` 両相）は spec に `sha256`（manifest の FileRef）と
`expectedBytes`（= `size`）を常時渡し、hub 自前の validate フック（毎ヒット全量 sha256 +
size 検査）を撤去する。miss 時は受信中に検証（不一致はエントリ不成立）、hit 時は記録ハッシュの
文字列比較、記録なしエントリは初回検証読みで backfill、記録不一致は自動 evict + refetch
（self-heal）。`recheck` は渡さない — opt-in/out の knob を設けない裁定は旧ドラフトから不変。
manifest 取得だけは従来どおり parse 検証のみ（事前の期待 sha256 が存在しない）。

**ADR 0070 決定 2 の「照合はキャッシュヒットにも走る（非交渉条件）」の帰結**: 条件の目的
（キャッシュ腐敗の検出）は記録ハッシュ側が引き継ぐが、検出できる腐敗の範囲は変わる —
**格納後のサイレントなビット腐敗は検出されない**（記録と中身が独立に腐る事故は文字列比較を
素通りする）。回復手段は `clearHubCache`（将来はキャッシュ保守波の `pruneHubCache`）。この
代償の受け入れは旧ドラフトで裁定済み。

### 2. キーはライブラリの内容キーをそのまま使う

hub はキーを合成しない。`sha256` を渡せば `["hf", kind, repo, path, sha256]`、渡さないもの
（manifest）は resolve URL キー。revision 跨ぎの同一バイトヒット・別 revision の共存はキー設計の
帰結として得る。旧ドラフトの `casKeyFor` / `karume-cache/2` 名前空間は不採用（不要になった）。

### 3. 認証別キャッシュ隔離の撤去（ユーザー裁定）

`cacheNameFor` の token 別名前空間（`karume/1:auth:<hex16>`）は復元しない。gated 資産の運用
予定がなく過剰防御（2026-08-28 裁定）。認証ヘッダは RequestInit として透過するが、キャッシュは
共有名前空間に載る — 同一端末で別 token の利用者が gated 資産の写しへ到達しうる制約は
limitations に by-design として記載する。将来 gated 運用を始める場合は custom `caches` ラッパ
（`open()` の remap）で隔離を再導入できる（0.5.0 README の移行節が示す代替）。

### 4. `AssetPhase` から `verifying` を撤去する

検証は fetch-cache 内部（受信中 / 文字列比較）に埋まり hub から観測できない — 観測できない
フェーズを偽装しない（breaking・pre-release）。`fileLoaded` / `fileTotal` は `complete` で
一致する契約のまま。

### 5. 移行と付随の獲得

- **旧名前空間の purge**: `loadManifest` 入口で 1 回、`karume/1` と `karume/1:auth:*` を
  `caches.delete`（失敗してもロードは続行）。キー形も変わるため旧エントリは miss して再 DL
  1 回（データ喪失ではない）。
- **`clearHubCache` の意味変更**: 対象が固定名前空間 `"fetch-cache"` になる — 同一 origin で
  アプリが fetch-cache を直接使っている場合はそれも消える（docstring に明記）。repo 単位の
  細粒度掃除（`evict(prefix)` / `listKeys`）はキャッシュ保守波の `pruneHubCache` で。
- **DL 前即エラーの獲得**: `expectedBytes` の確保失敗は受信前に throw（`cause` = RangeError）—
  「Chromium の単一 ArrayBuffer 上限超えを DL 後にしか観測できない」既知問題（backlog ③）が
  この追従で閉じる。hub は握りつぶさず最初の真因として上げる（取得失敗の真因復元機構と整合）。

## Consequences

- breaking（hub 0.7.0 系）: `AssetPhase` の縮小・`clearHubCache` の対象変更・（内部）検証の
  実体変更。anima-web は `verifying` フェーズの消滅に追随が要る。
- multi-GB 資産の 2 回目以降ロードから全量再ハッシュが消える（Pixel 級での起動 CPU/RAM 削減）。
- `sha256Hex` と直列化保険（Chrome の digest 全量コピー対策）は hub から消えるが、**全量コピー
  そのものが消えるのは逐次面の相 1 だけ**（prefetch は受信中の逐次ハッシュ）。全量面の network
  経路は fetch-cache が確定バイト列へ `crypto.subtle.digest` を呼ぶため Blink 側の +N ピークは
  残り、しかも hub からは直列化できない — 同時本数を抑えるのは `BYTE_BUDGET`（1.5GiB）のみに
  なった。初回 DL の RAM ピークは要実測（キャッシュヒットは記録比較だけで digest 自体が
  起きないため、2 回目以降は明確に改善側）。
- shard 配布波（R1）の相 1 prefetch は本 ADR の `sha256` 席が前提（0.5.0 の prefetch は記録を
  焼き、一致エントリを再 DL しない）。
