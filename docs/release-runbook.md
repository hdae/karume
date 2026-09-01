# リリース手順書（JSR + HF モデル公開）

> 性格: **恒久手順の正本**。リリース・モデル公開のたびに上から辿るチェックリスト。
> 機序・数値の正本は各所（断片化 =
> [research/2026-08-09-xet-fragmentation.md](research/2026-08-09-xet-fragmentation.md)・
> env の正本 = [assets-layout.md](assets-layout.md) 公開節・pin =
> ADR [0073](decisions/0073-models-source-pin.md)・publish 機構 =
> `.github/workflows/publish.yml`）。手順が実態と食い違ったら**この文書を先に直す**。

## 0. 順序の不変条件

**version bump → 配布形の焼き直し → HF アップロード → pin 焼き込み → JSR publish** の順
（非可換 MUST）。

- **焼き直しは bump の後**: manifest の `generator` 欄はパッケージ版を写す（`karume/<version>`）
  ので、bump 前に焼いた配布形は古い版を名乗ったまま HF へ載る。開発中のローカル再焼きは
  古い表記のままでよい（e2e が読むのはバイト列で generator ではない）が、**公開する 1 回は
  bump 後に焼き直す**。以下の節の並びは作業のまとまりであって時系列ではない — **§4 の
  lockstep bump コミットだけは §1〜§3 より前**に置く（残りの §4 = push / Release は最後）。
- pin の SHA はアップロードが済むまで存在しない（ADR 0073 — models の `*_CURRENT` は
  「公開時点の最新コミット」に固定する）。
- manifest format を上げた場合、旧 JSR クライアントは新リポを読めない（hub は単一形パース —
  ADR 0041/0071）。JSR publish を最後に置くことで「新 hub が読める資産が既に HF にある」
  状態で公開される。

### 越境参照を含むリポの公開順序（MUST）

越境コンポーネント参照（ADR [0038](decisions/0038-manifest-v1.md) §7 追記）は**参照先の
commit SHA を焼き込む**ので、参照先が先に公開されていないと焼けない。extra（追加学習系）が
公式リポの text stack を参照する現行の組（ADR 0087 — 旧 turbo → anima の向きと同型）では:

1. **`karume-anima` を先に上げる**（§2 の断片化対策込み・公式 3 変種 — ADR 0087）
2. その **main の commit SHA を確定**させる（§3 と同じ取り方）
3. その SHA を渡して **`karume-anima-extra` を越境参照で焼く** — `tools/export-recipes/dist.py`
   の 5 指定（`--ref-repo` / `--ref-revision` / `--ref-dist` / `--ref-model` / `--ref-role`）は
   **全部揃うか 1 つも無いか**の 2 通りだけで、部分指定は落ちる。
   **ステージングの `--out` は必ずリポ名と同名のディレクトリにする** — カードの Usage 例の
   repo 名は出力ディレクトリ名から導出されるため、別名で焼くと誤った repo 名がカードに載る
   （0.5.0 で `-release` 付きステージング名がそのまま公開カードに写った実害 — 2026-08-25 に
   修正。恒久策は backlog later）
4. **`karume-anima-extra` を上げる**

参照先を後から上げ直すと SHA が変わり、extra の manifest は**古い revision を指したまま**に
なる（バイト列は二重 pin で守られるので誤配は起きないが、2 リポの内容が別世代になる）—
**参照先を上げ直したら extra も焼き直して上げ直す**。

NOTE（次リリース限り）: 旧 `hdae/karume-anima-turbo` は ADR 0087 で退役 — 上げ直さない。
公開済みリポの扱い（deprecation 掲示・README 差し替え等）はアップロード時にユーザー裁定。

## 1. 事前検証

- [ ] `deno task verify` 緑（GPU テスト込み — アダプタ無し環境の SKIP はリリース判定では不可・
      ADR 0005）
- [ ] exporter: `uv run --no-sync pytest` + `uv run --no-sync ruff check` +
      `uv run --no-sync ruff format --check` を
      **tools/exporter と tools/export-recipes の両方**で緑（CI は lint と format を
      別ステップで回す — check だけ見て format を落とした実績 2026-08-21）
- [ ] 配布形の再生成が要る変更（manifest 形式・quant 席・カード・既定 quant）があったなら
      `models/` 配下の対象リポを dist で再生成し、各ファミリの gate（WAV / PNG / verify_dist）緑。
      **公開ぶんの焼き直しは §4 の bump コミットの後**（§0 の順序 — `generator` がパッケージ版を
      写す）。ローカル e2e の SBV2 門は公開ミラー `karume-sbv2-jvnv` が正本（2026-08-30 に
      非公開 fn から付け替え — fn ミラーは常設せず、必要時のみ assets-layout の dist コマンドで
      `inputs/sbv2/FN*` から再生成する）
- [ ] ライセンス確認: 各リポの `THIRD_PARTY_NOTICES.md` と生成カードの attribution が
      upstream の現物と一致（per-revision の人間確認 — release gate）
- [ ] git: 全てコミット済み・push はユーザー

## 2. HF アップロード（断片化対策 — MUST）

**背景**: env を付けずに上げると Xet の chunk dedup が効きすぎて再構成が断片化し、DL が
5〜6 倍遅くなる。断片化は一度 CAS に載ると戻らない（**片道ラチェット** — 初回で防ぐのが
唯一の手）。

```sh
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE=1000000
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE_HYSTERESIS_FACTOR=1.0
export HF_XET_DEDUPLICATION_NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1
```

- [ ] 上の env 3 本を**同一シェルで** export してから `hf upload` を実行する
      （正本: [assets-layout.md](assets-layout.md) 公開節）。**新規バイトにはこれで効く**
      （26〜30 MiB/term 実測）。
      **NOTE（2026-08-29 退行・hf_xet 1.4.3）**: 旧・4 本目
      （`GLOBAL_DEDUP_QUERY_ENABLED`）は**存在しない**（後継 =
      `HF_XET_MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES` を巨大値に — ただし下記には効かない）。
      さらに**リポ自身の履歴に同一 chunk がある場合の repo 内 dedup はどのノブでも止まらない**
      （[known-issues](known-issues.md) の text_encoder 断片化）。
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
      大きく下回っていたら断片化 — **現行クライアント（hf_xet 1.4.3）では回復手段が無い**。
      同バイト上げ直しは hf CLI が転送ごとスキップして効かず、delete→再 up の 2 コミット法でも
      不発（2026-08-29 実測 — 上の NOTE と同じ結論）。恒久対処の候補 3 案は
      [known-issues](known-issues.md)。観測値は §5 の記録へ残す

## 3. pin 焼き込み（ADR 0073）

- [ ] 公開した各リポの main の SHA（40hex）を取得:
      `curl -sS "https://huggingface.co/api/models/<owner>/<repo>/revision/main"` の `sha` 欄
- [ ] `packages/models` の pin 定数（`<FAMILY>[_<VARIANT>]_CURRENT` — ADR 0073 追記
      2026-08-25）の `revision` へ記入。**公開リポ 1 つにつき 1 定数**なので、上げたリポの
      定数を漏れなく: `ANIMA_CURRENT` / `SBV2_JVNV_CURRENT` / `IRODORI_V4_SMALL_CURRENT`
      （旧 `ANIMA_TURBO_CURRENT` は廃止 — ADR 0087。extra リポの pin 定数
      `ANIMA_EXTRA_CURRENT` は公開時に新設 = 下の項目）
- [ ] **初公開リポの pin 定数はこの時点で新設**（ADR 0073 決定 1 — 公開前に置くと 404 にしか
      ならない定数が公開面に生える。gemma4 の config.ts 冒頭 NOTE と同型）。次リリースで新設:
      `GEMMA4_CURRENT`・`IRODORI_V4_1_SMALL_CURRENT`・`ANIMA_EXTRA_CURRENT`
      （各 config.ts + `mod.ts` re-export + `current_source_test.ts` の CASES 追随）。あわせて
      `examples/irodori/main.ts` の台本既定を `IRODORI_V4_1_SMALL_CURRENT` へ切替
      （2026-09-01 裁定 — 旧 pin は温存）
- [ ] pin の更新は **bump のたびの義務**（`*_CURRENT` = 「このパッケージ版が**検証した**
      取得元」— ADR 0073 追記 2026-08-25）。下の疎通に加え、席や既定 quant が動いたなら
      動作テストまで通してから「検証した」と名乗る
- [ ] 疎通: pin 済み SHA での `fromPretrained` が実 URL で通ること（SHA 指定は revision 解決
      リクエストが発生しない = オフライン起動可 — ADR 0038）
- [ ] **pin 定数が公開 revision の唯一の在処**であること: `rg '\b[0-9a-f]{8,40}\b' docs
      .claude/ACTIVE_DESIGN.md` 相当で、docs 側へ SHA を写していないか確認する（写しは焼き直しの
      たびに古びる — 記録は `ANIMA_CURRENT` のような定数名で綴る。2 リリース連続で食い違った
      2026-08-24 / 08-29 の再発防止）
- [ ] `deno task verify` → コミット

## 4. JSR publish

機構: **GitHub Release を published にすると `publish.yml` が発火**し、workspace ルートの
`deno publish` が 3 パッケージ（hub / runtime → models）を依存順に一括 publish する
（既公開 version は冪等スキップ・OIDC トークンレス）。

- [ ] lockstep bump コミット: 3 JSR パッケージ + exporter の `version`、models → hub 等の
      `^` 依存、`deno.lock` の specifier、**`tools/uv.lock` の再生成（`uv lock` — MUST。CI は
      `uv run --locked` で鮮度検査するため、漏れると exporter / recipes 両ジョブが赤になる。
      実例: 0.4.3 の `e15e271`）**を揃えて 1 コミット（実績: `d65535c` / `7d97dd4`）
- [ ] push（ユーザー）→ CI 緑を確認（`ci.yml` は `deno publish --dry-run` で公開グラフも検証）
- [ ] CI 緑の main コミットから GitHub Release を作成 → published（発火）
- [ ] JSR 側で 3 パッケージの新 version を確認

PyPI `karume`（tools/exporter）は**未リリース**。公開を始める時にこの節へ手順を追記する。

## 5. 事後

- [ ] docs 同期: ACTIVE_DESIGN・backlog（release 節の消化状況）・リリース記録・
      プロジェクトメモリ
- [ ] 公開リポの実 DL 疎通: `fromPretrained(<FAMILY>_CURRENT)` を 1 回（`ref` に既定は無い —
      ADR 0073 追記 2026-08-25。文字列で渡すと `main` 追従になり pin の疎通確認にならない）
- [ ] 断片化検証の結果（§2）を research か backlog へ 1 行記録
