# リリース手順書（JSR + HF モデル公開）

> 性格: **恒久手順の正本**。リリース・モデル公開のたびに上から辿るチェックリスト。
> 機序・数値の正本は各所（断片化 =
> [research/2026-08-09-xet-fragmentation.md](research/2026-08-09-xet-fragmentation.md)・
> env の正本 = [assets-layout.md](assets-layout.md) 公開節・pin =
> ADR [0073](decisions/0073-models-source-pin.md)・publish 機構 =
> `.github/workflows/publish.yml`）。手順が実態と食い違ったら**この文書を先に直す**。

## 0. 順序の不変条件

**HF アップロード → pin 焼き込み → JSR publish** の順（非可換 MUST）。

- pin の SHA はアップロードが済むまで存在しない（ADR 0073 — models の既定ソース定数は
  「公開時点の最新コミット」に固定する）。
- manifest format を上げた場合、旧 JSR クライアントは新リポを読めない（hub は単一形パース —
  ADR 0041/0071）。JSR publish を最後に置くことで「新 hub が読める資産が既に HF にある」
  状態で公開される。

## 1. 事前検証

- [ ] `deno task verify` 緑（GPU テスト込み — アダプタ無し環境の SKIP はリリース判定では不可・
      ADR 0005）
- [ ] exporter: `uv run --no-sync pytest` + `uv run --no-sync ruff check` +
      `uv run --no-sync ruff format --check` を
      **tools/exporter と tools/export-recipes の両方**で緑（CI は lint と format を
      別ステップで回す — check だけ見て format を落とした実績 2026-08-21）
- [ ] 配布形の再生成が要る変更（manifest 形式・quant 席・カード・既定 quant）があったなら
      `models/` 配下の対象リポを dist で再生成し、各ファミリの gate（WAV / PNG / verify_dist）緑
- [ ] ライセンス確認: 各リポの `THIRD_PARTY_NOTICES.md` と生成カードの attribution が
      upstream の現物と一致（per-revision の人間確認 — release gate）
- [ ] git: 全てコミット済み・push はユーザー

## 2. HF アップロード（断片化対策 — MUST）

**背景**: env を付けずに上げると Xet の chunk dedup が効きすぎて再構成が断片化し、DL が
5〜6 倍遅くなる。断片化は一度 CAS に載ると既定設定の再アップロードでは戻らない
（**片道ラチェット** — 初回で防ぐのが唯一の低コストな手）。

```sh
export HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE=1000000
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE_HYSTERESIS_FACTOR=1.0
export HF_XET_DEDUPLICATION_NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1
```

- [ ] 上の env 4 本を**同一シェルで** export してから `hf upload` を実行する
      （正本: [assets-layout.md](assets-layout.md) 公開節）
- [ ] **書き込みトークンへ切替**: `hf auth switch --token-name "Karume Release"` —
      既定の読み取りトークン（Karume Gated Read）のままだと LFS batch が 403 になる
      （2026-08-21 実測）。アップロードが済んだら読み取りトークンへ戻す
- [ ] **再アップロード時**は先に `~/.cache/huggingface/xet/*/shard-cache` を退避する
      （断片化した祖先 shard がローカルに残っていると dedup ヒットで元に戻る）
- [ ] アップロード: `hf upload <owner>/<repo> models/<repo> . --repo-type model`
      （`models/` は 1 ディレクトリ = 1 HF リポ — assets-layout）

### アップロード直後の断片化検証（必須）

大きいファイル（重み safetensors）を代表 2〜3 本サンプルして reconstruction を確認する
（research §9 の再現手順）:

```sh
curl -sS -I "https://huggingface.co/<repo>/resolve/main/<path>" | grep -i x-xet-hash
curl -sS "https://huggingface.co/api/models/<repo>/xet-read-token/main"   # -> casUrl / accessToken
curl -sS -H "Authorization: Bearer <accessToken>" "<casUrl>/v1/reconstructions/<x-xet-hash>"
```

- [ ] `terms[]` を数え **MiB/レンジ ≥ 10 目安**（健全なら 1 xorb = 1 term に近い）。
      大きく下回っていたら断片化 — **同じバイト列のまま同じパスへ上げ直すと治る**
      （バイト不変なのでコミットは増えない。shard-cache 退避を忘れない）

## 3. pin 焼き込み（ADR 0073）

- [ ] 公開した各リポの main の SHA（40hex）を取得:
      `curl -sS "https://huggingface.co/api/models/<owner>/<repo>/revision/main"` の `sha` 欄
- [ ] `packages/models` の各ファミリ既定ソース定数（`*_DEFAULT_SOURCE` — ADR 0073）の
      `revision` へ記入
- [ ] anima の素版リポ（`hdae/karume-anima`）を上げ直したら `ANIMA_BASE_SOURCE` も同様に更新
      （既定ではないが pin の MUST は同じ）
- [ ] 疎通: pin 済み SHA での `fromPretrained` が実 URL で通ること（SHA 指定は revision 解決
      リクエストが発生しない = オフライン起動可 — ADR 0038）
- [ ] `deno task verify` → コミット

## 4. JSR publish

機構: **GitHub Release を published にすると `publish.yml` が発火**し、workspace ルートの
`deno publish` が 3 パッケージ（hub / runtime → models）を依存順に一括 publish する
（既公開 version は冪等スキップ・OIDC トークンレス）。

- [ ] lockstep bump コミット: 3 JSR パッケージ + exporter の `version`、models → hub 等の
      `^` 依存、`deno.lock` の specifier を揃えて 1 コミット（実績: `d65535c` / `7d97dd4`）
- [ ] push（ユーザー）→ CI 緑を確認（`ci.yml` は `deno publish --dry-run` で公開グラフも検証）
- [ ] CI 緑の main コミットから GitHub Release を作成 → published（発火）
- [ ] JSR 側で 3 パッケージの新 version を確認

PyPI `karume`（tools/exporter）は**未リリース**。公開を始める時にこの節へ手順を追記する。

## 5. 事後

- [ ] docs 同期: ACTIVE_DESIGN・backlog（release 節の消化状況）・リリース記録・
      プロジェクトメモリ
- [ ] 公開リポの実 DL 疎通: `fromPretrained("<owner>/<repo>")` を 1 回（既定 = pin 済み SHA）
- [ ] 断片化検証の結果（§2）を research か backlog へ 1 行記録
