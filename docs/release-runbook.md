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
  lockstep bump コミットと直後の venv 同期だけは §1〜§3 より前**に置く（残りの §4 = CHANGELOG の
  版の節への移動 / push / Release は最後）。
- pin の SHA はアップロードが済むまで存在しない（ADR 0073 — models の対応表
  `<FAMILY>_SOURCES` は「公開時点の最新コミット」に固定する。表の形は ADR
  [0092](decisions/0092-distribution-repos-and-sources.md) 決定 3）。
- manifest format を上げた場合、旧 JSR クライアントは新リポを読めない（hub は単一形パース —
  ADR 0041/0071）。JSR publish を最後に置くことで「新 hub が読める資産が既に HF にある」
  状態で公開される。

### 越境参照を含むリポの公開順序（MUST）

越境コンポーネント参照（ADR [0038](decisions/0038-manifest-v1.md) §7 追記）は**参照先の
commit SHA を焼き込む**ので、参照先が先に公開されていないと焼けない。extra（追加学習系）が
公式リポの text stack を参照する現行の組（ADR 0087 — 旧 turbo → anima の向きと同型）では:

1. **`karume-anima` を先に上げる**（§2 の断片化対策込み・公式 5 変種〈`anima-turbo-v1.1`〈既定〉/
   `anima-v1.0` / `anima-aesthetic-v1.1` / `anima-turbo-v1.0` / `anima-aesthetic-v1.0`〉— ADR 0087）
2. その **main の commit SHA を確定**させる（§3 と同じ取り方）
3. その SHA を渡して **`karume-anima-extra` を越境参照で焼く** — `tools/export-recipes/dist.py`
   の 5 指定（`--ref-repo` / `--ref-revision` / `--ref-dist` / `--ref-model` / `--ref-role`）は
   **全部揃うか 1 つも無いか**の 2 通りだけで、部分指定は落ちる。
   **extra は 2 モデル同居なので `--model anima-wai-v1.0 --model anima-copycat-20260610` で
   1 リポを 1 回で焼く**（`--model` は繰り返し指定 = ファミリー組み立て。先頭が `defaultModel`）。
   `--ref-role` は `text_encoder` / `vae_decoder` / `tokenizer` / `tokenizer_2` の 4 つ
   （`text_conditioner` は extra の 2 モデルとも自前なので越境しない — `distribution.py` の
   `own_text_conditioner=True`）。
   **カードの Usage 例に載る repo 名は pipeline の宣言（`Pipeline.repo_name`）から導かれ、`--out` の
   ディレクトリ名は見ない**（0.5.0 で `-release` 付きステージング名が公開カードに写った実害は、この導出で
   構造的に消えた）。extra は宣言の `karume-anima-extra` がそのまま載るので `--repo` は要らない。
   `--repo` は宣言を上書きする口で、引数は `OWNER/NAME` の形（例 `hdae/karume-anima-extra` — 名前だけを
   渡すと形の検査で落ちる）。`--repo` を付けたら `--out` も必須。複数モデルを 1 リポに束ねてモデルごとの
   宣言が揃わない系列では `--repo` が必須で、無ければ組み立て前に落ちる — 現行では `sbv2` の JVNV 4 声
   （宣言は話者ごとの `karume-sbv2-<話者>` なので `--repo hdae/karume-sbv2-jvnv` — 2026-09-24 に
   `resolve_repo` で確認）。
   `--out` は `models/<repo>`（リポ名と同名）に置く — §2 の台本は `models/<repo>` をそのまま上げる
4. **`karume-anima-extra` を上げる**
   - [ ] extra ミラーの生成後に**越境の実資産門を復活**させる — `packages/models/tests/e2e_anima_test.ts`
         の `CROSS_REPO_MIRRORS` と `packages/runtime/tests/assets_fusion_counts_test.ts` の
         `MIRRORS` にエントリを戻し、extra 変種の融合ヒット数と参照 sha を新規凍結する
         （公式リポの自己完結化で一旦空にした門 — 表と配り分けの機構は残してある）

参照先を後から上げ直すと SHA が変わり、extra の manifest は**古い revision を指したまま**に
なる（バイト列は二重 pin で守られるので誤配は起きないが、2 リポの内容が別世代になる）—
**参照先を上げ直したら extra も焼き直して上げ直す**。

NOTE（次リリース限り）: 旧 `hdae/karume-anima-turbo` は ADR 0087 で退役 — 上げ直さない。
公開済みリポの扱い（deprecation 掲示・README 差し替え等）はアップロード時にユーザー裁定。

## 1. 事前検証

- [ ] `deno task verify` 緑（GPU テスト込み — アダプタ無し環境の SKIP はリリース判定では不可・
      ADR 0005）。**リリース判定機の条件**: アダプタが `shader-f16` と `timestamp-query` を
      **列挙する**実 HW であること（ソフトウェアアダプタ〈lavapipe 等〉は f16 を f32 で計算する
      ので実 HW レーンと区別する）と、実重み系列（`outputs/series/`）が置いてあること。
      列挙・資産・参照値・配布形ミラーが欠けた機で verify を通すには意図表明の env
      （`KARUME_ALLOW_NO_SHADER_F16` / `KARUME_ALLOW_NO_TIMESTAMP_QUERY` /
      `KARUME_ALLOW_NO_ASSETS` / `KARUME_ALLOW_NO_REFERENCE` / `KARUME_ALLOW_NO_DISTRIBUTION`）が
      要るが、**それらを設定した環境の緑はリリース判定に使わない**（既定では `gpu_gate_test.ts` /
      `assets_gate_test.ts` / 系列ごとの参照門 / `distribution_gate_test.ts` が FAIL にする）
- [ ] **期待値を作るモードが 1 つも立っていないこと**: `KARUME_REFERENCE`（`write` / `rewrite`）と
      `KARUME_SURFACE=write` は期待値を実測で書き替える口で、検証ではない。とくに参照門の緑条件は
      「作るモードなら行が 0 件でも緑」（`packages/runtime/tests/helpers/reference.ts`）なので、
      リリース判定機に `KARUME_REFERENCE` を残すと **sha 門が 1 本も突き合わせないまま全て
      `written` で緑になる**。両方とも未設定で回した緑だけをリリース判定に使う
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
- [ ] ライセンス確認: 各 family の provenance 記録
      （`tools/export-recipes/<family>/THIRD_PARTY_NOTICES.md`）と、配布リポの生成カードの
      attribution が upstream の現物と一致（per-revision の人間確認 — release gate。正本 =
      backlog release 節。配布リポ直下に置けるのは `LICENSE.md` / `NOTICE.md` だけ —
      `verify_dist` の `LEGAL_PATHS`）
- [ ] git: 全てコミット済み・push はユーザー

## 2. HF アップロード（断片化対策 — MUST）

**背景**: env を付けずに上げると Xet の chunk dedup が効きすぎて再構成が断片化し、DL が
5〜6 倍遅くなる。dedup には **global dedup**（CAS 全体への chunk 照会 — 自分が上げていない
他人の xorb にもヒットする）が含まれ、これを止めないと**初回アップロードでも**断片化する。

- [ ] **使う `hf` を固定する**: `tools/.venv/bin/hf`（huggingface_hub 1.27 / hf_xet 1.6.0）。
      **nix の `hf`（hf_xet 1.4.3）は使わない** — global dedup の停止ノブ（下の 4 本目）が
      その版には無く、断片化を止める手段が無い

```sh
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE=1000000
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE_HYSTERESIS_FACTOR=1.0
export HF_XET_DEDUPLICATION_NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1
export HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false
```

- [ ] 上の env 4 本を**同一シェルで** export してから `hf upload` を実行する
      （正本: [assets-layout.md](assets-layout.md) 公開節）。**台本
      `tools/release/hf-upload.zsh upload <repo>` がこの env・shard-cache 退避・venv の hf・
      直後の断片化表までを 1 回で行う** — 手で打つのは台本が使えないときだけ。表には
      `### FAILED` 行（xet のハッシュか再構成を読めなかった part — 台本は非 0 で終わる。原因を解消してから
      上げ直す）と `### SKIP` 行（xet 以外に格納された小さいファイル）も出る（2026-09-25）。4 本目が
      読まれたことは hf_xet のログの `global_dedup_query_enabled = false (user set)` 行で
      確認する（台本はこの行と CAS 照会回数をログへ写す）。
      **NOTE（hf_xet の版差）**: 4 本目 `GLOBAL_DEDUP_QUERY_ENABLED` は
      **1.4.3 には無く 1.6.0 にはある**（2026-09-04 にバイナリの env 名一覧で確認）。
      1.4.3 で後継とされた `HF_XET_MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES` を巨大値にする形は
      **効かない**（2026-08-29 実測）。さらに 1.4.3 では**リポ自身の履歴に同一 chunk がある
      場合の repo 内 dedup がどのノブでも止まらない**（同日 anima の text_encoder で実測。
      shard 分割後の 0.8.0 では再現せず — 結果は
      [退避した消化済み節](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)の 0.8.0 節）。
- [ ] **`~/.cache/huggingface/xet/*/shard-cache` を毎回退避する**（再アップロードだけでなく
      **初回でも**）。global dedup のヒットでサーバから取り寄せた shard がここに残り、
      次のアップロードはそれを引き当てて断片化を継承する
- [ ] **書き込みトークンへ切替**: `tools/.venv/bin/hf auth switch --token-name "Karume Release"` —
      既定の読み取りトークン（Karume Gated Read）のままだと LFS batch が 403 になる
      （2026-08-21 実測）。**アップロードが済んだら読み取りトークンへ戻す**:
      `tools/.venv/bin/hf auth switch --token-name "Karume Gated Read"` → `tools/.venv/bin/hf auth list` で選ばれている
      トークンを確かめる。NOTE: この戻しは守られていない実態がある（2026-09-24 時点で 09-23 の
      アップロード以来 Karume Release が選ばれたまま）。書き込みトークンを常用すると、読むだけの作業でも
      誤操作がそのまま HF への書き込みになる
- [ ] **台本は機械の門も兼ねる**（人の目視に依存しない）: `upload` は shard-cache の退避に
      失敗した時点で非 0 終了し、アップロード後に `global_dedup_query_enabled = false` と
      CAS 照会 0 回を検査して不一致なら非 0 終了する。`check` は `models/<repo>` に
      `.krm` / `.safetensors` が 1 本も無ければ非 0 で落ちる（空表を「検証したが問題なし」として出さない）
- [ ] アップロード: `tools/release/hf-upload.zsh upload <repo>`（中身は
      `tools/.venv/bin/hf upload hdae/<repo> models/<repo> . --repo-type model` —
      `models/` は 1 ディレクトリ = 1 HF リポ — assets-layout。追加引数はそのまま hf へ渡る）
- [ ] **配布形を変えた回（`karume/4` → `karume/5` など）は、新しい manifest が参照しない旧形のファイルを
      同じアップロードで消す**: `tools/release/hf-upload.zsh upload <repo> --delete '*.safetensors'`。
      `hf upload` は手元に無いファイルを消さないので、指定しないと旧形のファイルが main に残る。
      glob は `/` をまたいで入れ子の path にも当たる。同じ呼び出しで上げ直す path は削除から外れるので、
      sbv2 の `speakers/` / `styles/` のような資産の safetensors は残る。削除はアップロードの最初のコミットに
      乗り、大きいリポは複数コミット（`(part N)` — huggingface_hub のコミット分割の番号で、容器の part とは
      別物）に分かれる。どれも huggingface_hub 1.27 の `upload_folder` の実装の振る舞いで、仕様の保証ではない
      — 版を上げたら確かめ直す
- [ ] **上げる直前に HF の main とローカルミラーを突き合わせ、HF にだけ在る path を `--delete` に足す**:
      `curl -sS "https://huggingface.co/api/models/hdae/<repo>/tree/main?recursive=true"` の `type: "file"` の
      `path` 一覧と、`models/<repo>` 配下のファイル一覧の差を取る。`'*.safetensors'` だけでは旧形の
      safetensors 以外が残る（例: `karume-gemma4` の旧 PLE 索引 `e2b/ple/ple.json`）。`--delete` は
      繰り返し指定できる — 例 `tools/release/hf-upload.zsh upload karume-gemma4 --delete '*.safetensors'
      --delete 'e2b/ple/*'`。NOTE（推測）: tree API は大きな一覧をページ分割して返しうる（huggingface_hub の
      `list_repo_tree` は同じ endpoint を `Link` ヘッダで辿る実装）ので、1 回の curl では一覧が欠ける可能性が
      ある — 件数が多いリポは `tools/.venv/bin/python -c` で `HfApi().list_repo_files("hdae/<repo>")` を
      使う。HF の既定ファイル `.gitattributes` は差に出るが消さない。アップロード後に同じ
      突き合わせをもう一度行い、差が `.gitattributes` だけであることを確かめる
- [ ] **pin は削除後の main で焼く**: §3 の SHA はアップロードの全コミットが済んだ後の main
      （台本のログの `### main sha` 行）から取る。削除を別の回に分けると main が動き、pin を 2 回付け替える
      ことになる（実例: irodori-v4.1-small は旧 safetensors 65 本を後から消して pin を付け替えた — `1c952948`）。
      旧 revision のファイルは HF の履歴に残り、公開済みの旧版パッケージは旧 pin から読み続ける
- [ ] **リポ名の改名（該当回のみ）**: `tools/.venv/bin/hf repos move <old> <new>` で改める。旧名は API /
      resolve とも **HTTP 307** で新名へリダイレクトするので既公開の参照は切れないが、
      **リダイレクトが生きていることを実際に叩いて確認**する —
      `curl -sS -o /dev/null -w '%{http_code} %{redirect_url}' "https://huggingface.co/api/models/<old>"`
      が `307` と新名の URL を返せばよい。
      改名後はモデルカードと `karume.json` を焼き直して上げ直す（カードの Usage の repo 名が
      旧名のまま残るため）ので **revision が動く** — §3 の対応表を新名 + 新 SHA で更新する

### アップロード直後の断片化検証（必須）

**全 krm / safetensors を表にする**（代表 2〜3 本のサンプルでは足りない — 2026-09-04 の siglip2 は
so400m の 7 shard 中 5 本だけが断片化しており、サンプルの当たり外れで見落とす）。
`tools/release/hf-upload.zsh upload` は終了時に表を出す。公開済みリポを後から見るときは
`tools/release/hf-upload.zsh check <repo>`。表の中身は research §9 の再現手順そのもの:

```sh
curl -sS -I "https://huggingface.co/<repo>/resolve/main/<path>" | grep -i x-xet-hash
curl -sS "https://huggingface.co/api/models/<repo>/xet-read-token/main"   # -> casUrl / accessToken
curl -sS -H "Authorization: Bearer <accessToken>" "<casUrl>/v1/reconstructions/<x-xet-hash>"
```

- [ ] `terms[]` を数え **MiB/レンジ ≥ 10 目安**（健全なら 1 xorb = 1 term に近い）。
      大きく下回っていたら断片化 — **hf_xet 1.6.0 + 上の env 4 本なら回復できる**。
      手順 = shard-cache を退避 → **リポを削除して再作成** → 同一バイトを上げ直す。CAS 照会が
      0 回になり xorb を新規に書くので健全形に戻る（2026-09-04・siglip2 で 46〜61 MiB/term へ
      回復 — [research](research/2026-08-09-xet-fragmentation.md) の 2026-09-04 追記）。
      同一リポ内の delete → 再 up の 2 コミット法は**効かない**（2026-09-05 実測 — 同一バイトは転送されず元の
      xorb を参照したまま）。同一 checkpoint の 2 解像度を同居させたリポでは削除 → 再作成でも 2048 側の shard
      1 本が回復しなかった（[research 2026-08-09 の 2026-09-05 追記](research/2026-08-09-xet-fragmentation.md)）。
      **MUST NOT: 公開 pin のあるリポでは削除 → 再作成を使わない**（ADR 0108 決定 18 — HF リポの履歴は残し、
      公開済みの旧版パッケージは旧 revision の pin から読み続ける）。リポを削除すると旧 pin が 404 になるという
      見立ては**推測**（実機では確かめていない）。公開 pin のあるリポで断片化したら、削除せずに
      止めて裁定を仰ぐ。
      **hf_xet 1.4.3 では回復手段が無い**（片道ラチェット — 同バイト上げ直しは hf CLI が転送ごと
      スキップし、2 コミット法も不発。2026-08-29 実測）。観測値は §5 の記録へ残す

## 3. pin 焼き込み（対応表への記入 — ADR 0073 / 0092）

在処は `packages/models` の家族別対応表 `<FAMILY>_SOURCES`（キー = HF リポ名の basename から
`karume-` を落としたもの）と、barrel が全家族を畳んだ `KARUME_SOURCES`（ADR
[0092](decisions/0092-distribution-repos-and-sources.md) 決定 3）。

- [ ] 公開した各リポの main の SHA（40hex）を取得:
      `curl -sS "https://huggingface.co/api/models/<owner>/<repo>/revision/main"` の `sha` 欄
- [ ] 上げたリポの**該当キーの `revision` へ記入**する
      （例: `IRODORI_SOURCES["irodori-v4.1-small"].revision`）。キーはリポ名から機械的に決まる
      ので、リポ名を見れば書き先が一意に決まる
- [ ] **`KARUME_SOURCES` の網羅を確認**: 今回上げたリポが全て表に載っていること。
      不変条件（`"karume-" + key === repo の basename`・owner `hdae`・`revision` は 40hex）は
      テストが門になっているので、`deno task test` で落ちる
- [ ] **初公開リポのエントリはこの時点で新設**（ADR 0073 決定 1 の理由を継承 — 公開前に置くと
      404 にしかならないキーが公開面に生える）。手順は各 family の `config.ts` の
      `<FAMILY>_SOURCES` へ 1 エントリ足すだけで、barrel の re-export は表単位なので追随不要
- [ ] pin の更新は **bump のたびの義務**（対応表 = 「このパッケージ版が**検証した**取得元」—
      ADR 0073 追記 2026-08-25 の維持義務を継承）。下の疎通に加え、席や既定 quant が動いたなら
      動作テストまで通してから「検証した」と名乗る
- [ ] 疎通: 記入した SHA での `fromPretrained` が実 URL で通ること（SHA 指定は revision 解決
      リクエストが発生しない = オフライン起動可 — ADR 0038）。**表の全エントリを横断する疎通は
      §5 の `deno task smoke:published`** — 公開版 `@karume/hub` で `KARUME_SOURCES` を総なめ
      するので、JSR publish 後にしか打てない（§0 の順序）
- [ ] **対応表の値が公開 revision の唯一の在処**であること: `rg '\b[0-9a-f]{8,40}\b' docs
      .claude/ACTIVE_DESIGN.md` 相当で、docs 側へ SHA を写していないか確認する（写しは焼き直しの
      たびに古びる — 記録は `ANIMA_SOURCES["anima"]` のようなキーで綴る。2 リリース連続で
      食い違った 2026-08-24 / 08-29 の再発防止）
- [ ] `deno task verify` → コミット

## 4. JSR publish

機構: **GitHub Release を published にすると `publish.yml` が発火**し、workspace ルートの
`deno publish` が 3 パッケージ（hub / runtime → models）を依存順に一括 publish する
（既公開 version は冪等スキップ・OIDC トークンレス）。

NOTE: パッケージ README / LICENSE は公開物に含まれる（`packages/*/README.md` と
`packages/*/LICENSE` — `deno publish --dry-run --allow-dirty` の出力に 3 パッケージとも載ることを
2026-09-20 に確認。各 `deno.json` の `publish` は `exclude: ["tests/"]` だけで、`include` の明示は
要らない）。

- [ ] lockstep bump コミット: 3 JSR パッケージ + exporter の `version`、models → hub 等の
      `^` 依存、`deno.lock` の specifier、**`tools/uv.lock` の再生成（`uv lock` — MUST。CI は
      `uv run --locked` で鮮度検査するため、漏れると exporter / recipes 両ジョブが赤になる。
      実例: 0.4.3 の `e15e271`）**を揃えて 1 コミット（実績: `d65535c` / `7d97dd4`）。
      `CHANGELOG.md` はこのコミットに含めない（下の項目 — bump は §1〜§3 より前に切るので、
      焼き直しからアップロードまでの間の変更も `[Unreleased]` へ書き続ける）
- [ ] **bump の直後に venv を同期する（MUST）**: `(cd tools && uv sync --all-groups)`。manifest の
      `generator` 欄は pyproject でも uv.lock でもなく**venv に入っている `karume` の dist-info の版**を
      写す（`karume.dist.generator_tag()` = `importlib.metadata.version`）ので、`uv lock` だけで
      `--no-sync` の `uv run` を続けると旧版を名乗ったまま焼ける。機械確認 = 焼き直した各 `dist.py` の
      最終行 `[dist] <out> — karume/<版> / ...` の `<版>` が bump 後の版であること
- [ ] **リリース直前の別コミット**で `CHANGELOG.md` の `[Unreleased]` をその版の節へ移し、見出しに
      日付（そのコミットの `git log -1 --format=%cs` と同じ `YYYY-MM-DD`）と版へのリンク定義を入れる。
      §3 の pin コミットの後・push の前に置く（ADR [0008](decisions/0008-public-api.md) の「breaking は
      CHANGELOG で明示する」の実体 — Release を作るコミットに版の節が無いと公開タグの中身と食い違う）
- [ ] push（ユーザー）→ CI 緑を確認（`ci.yml` は `deno publish --dry-run` で公開グラフも検証）
- [ ] CI 緑の main コミットから GitHub Release を作成 → published（発火）
- [ ] JSR 側で 3 パッケージの新 version を確認

PyPI `karume`（tools/exporter）は**未リリース**。公開を始める時にこの節へ手順を追記する。

## 5. 事後

- [ ] docs 同期: ACTIVE_DESIGN の Now・backlog（now 見出しと release 節の消化状況）・
      リリース記録・プロジェクトメモリ。**この項目が済むまでリリースは「完了」ではない** —
      台帳の同期はここ 1 か所を唯一の門として扱う（0.9.0 では backlog だけが更新され
      ACTIVE_DESIGN が「publish はユーザー」のまま残った。同型の漏れは 2 度目）
- [ ] 公開パッケージからの疎通: `deno task smoke:published`（tools/published-smoke — 公開版
      `@karume/hub` で `KARUME_SOURCES` 全エントリの manifest を解決し、sbv2 を
      `fromPretrained` まで通す。GPU が
      無い機体は `--manifests-only`）。ワークスペース配下の `jsr:@karume/*` はローカル member に
      解決されるため、疎通は必ずこの task で打つ（自前の deno.json で registry を引く）。
      **公開したての依存は `minimumDependencyAge` に引っかかる** — `tools/published-smoke/deno.json`
      の `exclude` に `jsr:@karume/*` と hub が引く `jsr:@hdae/fetch-cache` を載せてある（0.11.0 で
      公開 1 日の fetch-cache 0.7.0 が止まった実例・2026-09-06）。新しい `@hdae/*` 依存を足したら
      ここにも足す
- [ ] 断片化検証の結果（§2）を research か backlog へ 1 行記録
