# JSR publish の CI/Release 設計（monorepo・OIDC・ロックステップ版管理）

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。
> JSR / Deno の仕様は 2026-08-05 時点の一次ソース（docs.deno.com・jsr-io/jsr・実リポの workflow）に基づく。

2026-08-05・公開準備。3 パッケージ（`@karume/runtime` / `@karume/hub` / `@karume/models`）を
JSR へ公開するための CI/Release を設計・実装した記録。ユーザー承認 = トリガは GitHub Release
駆動（A1）・版管理はロックステップ（B1）。

## 前提となる調査結論（一次ソース）

- **monorepo は正式サポート**: `deno publish` を workspace ルートで実行すると、`name`+`exports`
  +`version` を持つ全メンバーを**依存順（hub/runtime → models）に一括 publish** し、公開コード内
  の workspace 参照を正規の `jsr:` 参照へ**自動書き換え**する（手書き不要）。初回（依存先が JSR
  未公開）でも同一実行内で依存先が先に publish されるため通る。出典: docs.deno.com/runtime/
  fundamentals/workspaces（2026-06-29 更新）・Deno in 2024 ブログ。メンバー除外は当該 `deno.json`
  に `"publish": false`。
- **OIDC トークンレス publish は GitHub Actions 限定**: `permissions: { contents: read,
  id-token: write }` のみで secret 不要。**事前に各パッケージの settings で GitHub リポを Link**
  する必要がある（未リンクは `actorNotAuthorized`）。他 CI はトークン方式（`--token`）。
  provenance（SLSA・Sigstore Rekor）は OIDC 経由のときだけ自動生成される。出典: jsr-io/jsr
  frontend/docs/publishing-packages.md・trust.md。
- **`deno publish --dry-run` は認証不要で PR ゲートに置ける**: slow types と公開 API 型出力を
  検証（実 publish はしない）。GHA での dry-run 認証バグは denoland/deno PR #22679 で修正済み。
  限界: `jsr:` の version 制約は検査しない（#22835・修正済みだが dry-run と実 publish の差は残る
  前提で扱う）。module graph 構築のため未キャッシュの外部依存取得でネットに触れ得る。
- **初回ブートストラップ（Web 手作業・CI では不可）**: `jsr.io/new` でスコープ `@karume` と 3
  パッケージを作成 → 各 settings で `hdae/karume` を Link。OIDC publish は既存パッケージ前提。

### 実在 monorepo の CI 実例（設計の裏取り）

| リポ                 | トリガ                                     | publish 前ゲート                                                | 版管理                           | タグ                                  |
| -------------------- | ------------------------------------------ | --------------------------------------------------------------- | -------------------------------- | ------------------------------------- |
| denoland/std         | `release: published` + `workflow_dispatch` | `deno fmt --check` → `deno task test`                           | 独立 + `deno bump-version`       | `release-YYYY.MM.DD`（日付・横断1本） |
| c4spar/deno-cliffy   | `pull_request` / `push` / `release`        | PR/push は `deno publish --dry-run` のみ（テストは別 workflow） | ロックステップ（全メンバ同一版） | `vX.Y.Z`                              |
| freshframework/fresh | `push: main` 毎                            | 無（冪等スキップ頼み）                                          | 独立                             | なし                                  |

karume は「実 GPU テストが CI で回らない」制約を持つため、**cliffy 型（publish ゲートは静的検査と
dry-run、実テストは別ルート）**を骨格に、トリガは std 型の Release 駆動を採る。

## 採用した構成

### `.github/workflows/ci.yml`（PR + main push）

`deno fmt --check` / `deno lint` / `deno check`（3 mod.ts）/ `deno test -A`（`KARUME_ALLOW_NO_GPU=1`
で GPU ケースを SKIP・非 GPU の回帰を捕捉）/ `deno publish --dry-run`。Deno は 2.9.4 にピン
（ローカル開発機と一致。fmt の挙動はバージョン差で割れ得る）。

### `.github/workflows/publish.yml`（Release published + workflow_dispatch）

`permissions: { contents: read, id-token: write }` で `deno publish`（workspace ルート・3
パッケージ一括）。version が既公開なら冪等スキップ。`workflow_dispatch` は初回・緊急時の手動用。

**テストゲートを publish.yml に置かない理由**: Release は CI 緑の main コミットから切る運用で、
テストは ci.yml に集約済み。`deno publish` 自体も slow types 等を検証し通らなければ publish
しないため、二重化しない。

## Release 運用手順（ロックステップ B1）

版管理は 3 パッケージを常に同一 version で揃える（開発初期・相互依存が密なため「一つ覚えれば
済む」）。手順:

1. **bump**: workspace ルートで `deno bump-version <patch|minor|major>` を実行する。
   - 3 メンバーの `version` を**同一 increment**で更新し、`packages/models/deno.json` の
     `jsr:@karume/hub` / `jsr:@karume/runtime` の**制約も自動で追随書き換え**する（実測: minor で
     `^0.1.0` → `^0.2.0`）。B1 が壊れない核心はこの自動書き換え。
   - `--dry-run` で書き込まず差分だけ確認できる。**experimental 扱い**（`deno bump-version is
     experimental and subject to change` の警告が出る — 挙動が変わったらこの手順を見直す）。
   - increment を省くと Conventional Commits からの個別バンプ + `Releases.md` 追記モードになる
     が、B1 では**必ず increment を明示**する（個別バンプは B2＝独立版管理の挙動）。
2. **commit**: `chore(release): vX.Y.Z` で 3 つの `deno.json` を 1 コミット。
3. **tag + Release**: `vX.Y.Z` タグを打ち、GitHub Release を作成して **publish**（draft のままだと
   発火しない）。これが `publish.yml` の `release: published` を発火させる。
4. CI（publish.yml）が OIDC で 3 パッケージを依存順に公開する。version が既公開なら no-op。

初回のみ: 上記の前に Web でスコープ + 3 パッケージ作成 + リポ Link（前述）。初回公開は
`workflow_dispatch` の手動実行でも代替できる。

## fetch 失敗・未確定（正直な限界）

- `jsr.io/docs/*` は WebFetch が 403（Cloudflare bot 遮断）。JSR 公式ドキュメントは GitHub 原本
  `jsr-io/jsr/frontend/docs/*` を一次代替として読んだ。運営側の version immutability 明文規定は
  別ルート未確認。
- `deno bump-version` の workspace メンバー imports 書き換えは**本リポで実測確認済み**（上記）だが、
  experimental のため将来のバージョンで挙動が変わり得る。
- exporter（PyPI `karume`）の CI は本設計のスコープ外（別タスク）。
