# 資産の置き場（models / outputs / inputs）

ローカル資産 3 根の規約（outputs の 4 分割は 2026-08-30 裁定）。綴りの正本は
`tools/export-recipes/_shared/paths.py`（`DIST_ROOT` / `SERIES_ROOT` / `EXAMPLES_ROOT` /
`BENCH_ROOT` / `MISC_ROOT` / `INPUTS_ROOT` / `OUTPUTS_ROOT`）。**3 根とも git 追跡しない**
（全て再生成可能な生成物か手置きの実重みで、リポジトリが持つのは作り方だけ）。

| 根                                     | 中身                                                               | 例                                                        |
| -------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------- |
| `models/`                              | **配布形だけ**（1 ディレクトリ = 1 HF リポ・そのまま上げられる）   | `models/karume-anima-turbo/` / `models/karume-sbv2-jvnv/` |
| `outputs/series/`                      | exporter の系列出力（コンテナ + golden フィクスチャ `io.*`）       | `outputs/series/sbv2-F1-f16/`                             |
| `outputs/examples/<model>/`            | examples 台本の既定出力先（`<model>` = `--source` の basename）    | `outputs/examples/karume-sbv2-jvnv/*.wav`                 |
| `outputs/bench/<model>/<日付>_<目的>/` | e2e ダンプ・ベンチ・視認評価（**消して安全** — 旧 `demo/` の後継） | `outputs/bench/karume-anima/2026-08-30_e2e-mismatch/`     |
| `outputs/misc/<名前>/`                 | ホスト資産（**消すと再取得・再エミットが要る**）                   | `outputs/misc/sbv2-demo/` / `outputs/misc/corpus/`        |
| `inputs/<family>/<name>/`              | 手置きの実重み（ckpt・config — 生成物ではない）                    | `inputs/sbv2/F1/`                                         |

- 系列出力にはコンテナ以外の**ホスト側資産**も入る（グラフを持たない compile 生成物）—
  トークナイザは `<系列名>-tokenizer/tokenizer.json`（例
  `outputs/series/gemma4-e2b-tokenizer/`）、anima のデモ用表は `anima-demo/text/`。export 系列の
  ディレクトリへは混ぜない（`dist` の宣言外ファイル検査が拾う）。
- **例外は同一コンポーネントの sidecar** — gemma4 製品系列の PLE（`ple.json` /
  `ple-NNNNN-of-NNNNN.safetensors` / `ple.probe.safetensors`）は系列ディレクトリの**中**に置く。
  トークナイザと違って独立コンポーネントではなく、製品グラフと**同じ配布 digest set の一員**
  （ADR [0085](decisions/0085-ple-host-gather.md) Consequences / ADR
  [0084](decisions/0084-gemma-tokenizer-chat.md) 決定 5）なので、別ディレクトリに据えると
  「新しいグラフ + 古い PLE」の組が作れてしまう（据え替えは系列ディレクトリごと 1 回）。
- `bench/` の `<日付>_<目的>` は実行日 YYYY-MM-DD + 短い識別スラグ（`e2e-mismatch` /
  `eval-images` / `quant-sim` 等）。ファイル取り違え防止のための規約で、機械（テスト・台本）も
  この形で書く。
- `outputs/misc/corpus/` は**テスト入力の凍結コピー**（実画像 4 枚 = depth-anything / birefnet /
  siglip2 の実画像門・golden 生成の入力。実音声 `vowel-*.wav` も同様）。正本の生成は
  `examples/anima/eval-images.ts` / `examples/irodori/eval-audio.ts`（bench へ出る）で、採用分を
  **人手でここへコピー**する — 生成先と凍結先を分けることで、ベンチ再実行がコーパスを黙って
  上書きする事故を断つ。
- `outputs/misc/yomi/` の日本語辞書（`*.jtd`）の取得: HF dataset `hdae/yomi-dict` の
  `naist-jdic.jtd.gz` を解いて置く（無いと models の修正辞書テストは SKIP される）。
- 旧 `outputs/series-archive/`（裁定済み系列の退避先）は 2026-08-30 の掃除裁定で**廃止**
  （base i4 の復活レバーは `outputs/series/` の `*-i4-dyn` 系列が引き続き担う —
  `anima/distribution.py` の NOTE）。

## 組み立て（系列 → 配布形）

```sh
cd tools/export-recipes
uv run python dist.py --model anima-v1.0 --model anima-wai-v1.0 \
    --model anima-copycat-20260610 \
    --out ../../models/karume-anima                  # 素の base 系（多 step + CFG）
uv run python dist.py --pipeline anima-turbo         # → models/karume-anima-turbo/（LoRA 焼き込み）
uv run python dist.py --pipeline irodori             # → models/karume-irodori-v4-small/
uv run python dist.py --pipeline sbv2 --card-profile jvnv \
    --model F1 --model F2 --model M1 --model M2 --out ../../models/karume-sbv2-jvnv
uv run python dist.py --pipeline sbv2 --card-profile fn \
    --model FN1 --model FN2 --model FN3 --model FN4 --model FN5 --model FN6 \
    --model FN7 --model FN8 --model FN9 --model FN10 --out ../../models/karume-sbv2-fn
uv run python dist.py --pipeline gemma4              # → models/karume-gemma4-e2b/（約 4.0GiB）
```

- `karume-gemma4-e2b` は**系列 2 本**（`gemma4-e2b-product` の製品コンテナ + PLE sidecar と
  `gemma4-e2b-tokenizer` の compile 済み資産）を 1 リポへ畳む。PLE sidecar は `assets` の席に載り、
  **asset 名は `ple.json` が書いた shard のファイル名そのもの**（読み手が索引 1 本で取得キーも
  引けるようにするため — 詳細は `tools/export-recipes/gemma4/README.md`）。上流が Apache 2.0 なので
  リポ直下に `LICENSE.md` / `NOTICE.md` が入る（`karume.dist` の法的テキスト席）。**未公開**なので
  pin 定数（`*_CURRENT`）はまだ無い（ADR [0073](decisions/0073-models-source-pin.md) 決定 1）。

- `karume-sbv2-fn` のミラーは**常設しない**（2026-08-30 裁定 — e2e の門はライセンス記述が正の
  `karume-sbv2-jvnv` を正本にする）。上のコマンドは再生成方法の記録で、系列（`inputs/sbv2/FN*`
  からの export → `outputs/series/`）が揃っていればいつでも焼き直せる。

- 仕様の正本は ADR [0041](decisions/0041-manifest-v2.md)（リポ内レイアウト = モデル別
  サブツリー + `shared/`・**配置は常に独立コピー** — ハードリンク禁止の理由も同 ADR 追記）+
  ADR [0071](decisions/0071-manifest-v3-shards.md)（shard 欄）+ ADR
  [0075](decisions/0075-quant-presentation.md)（quant の `label` / `description`・
  `requiredLimits`・ファイル参照の越境 `repo` / `revision`）。**現行 format は `karume/4`**
  （hub は単一形パース = `karume/4` 以外を読まない — `packages/hub/src/manifest.ts`）。
  `karume.json` は現物から導出（手書き禁止 — ADR 0038）。
- 組み立ては冪等（再実行で置き換え）。`verify_dist` が宣言と現物の突合・宣言外ファイル検査まで
  行い、モデルカード `README.md` は検証済み manifest から機械生成される（帰属は
  `--card-profile` — exporter の README 参照）。
- 系列を消しても配布形は壊れない（独立コピー）。逆に配布形は `dist.py` でいつでも系列から
  再生成できる。core 単体の `karume dist` は受理集合が空で落ちる設計（ADR 0065）— family を
  組むのは常にこのリポ driver の `dist.py`。

## 公開（HF へのアップロード）

**MUST: モデルファイルを HF へ上げるときは、以下の env を必ず付ける。**

```sh
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE=1000000
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE_HYSTERESIS_FACTOR=1.0
export HF_XET_DEDUPLICATION_NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1
```

付けないと Xet の chunk 単位 dedup が効きすぎて**再構成が断片化し、DL が 5〜6 倍遅くなる**。
断片化は一度 CAS に載ると戻せない（片道ラチェット）ので、**初回アップロードで防ぐのが唯一の手**
である。機序と実測は
[research/2026-08-09-xet-fragmentation.md](research/2026-08-09-xet-fragmentation.md)。

- 上げたら**必ず検証する** — reconstruction の term 数を数え、`MiB/レンジ` が 10 を大きく
  下回っていないか見る（手順は同ドキュメント §9）。健全なら 1 xorb = 1 term に近くなる。
- 再アップロードの前には `~/.cache/huggingface/xet/*/shard-cache` を退避する（断片化した祖先の
  shard がローカルに残っていると、そこへ dedup ヒットして元に戻る）。

**env が効く範囲は hf_xet の版に依存する。** 実測で確定している 2026-08-29 時点（hf_xet 1.4.3）
の線引き:

- **新規バイトのアップロードには依然として有効**（実測 26〜30 MiB/term = 健全）。上の 3 本は
  必ず付ける。
- 旧・4 本目の `HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED` は **1.4.3 で消滅**した
  （後継は `HF_XET_MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES` を巨大値にする形だが、下の
  repo 内 dedup には効かない）。存在しない env を export しても何も起きないので、
  「対策済み」と読まないこと。
- **リポ自身の履歴に同一 chunk がある場合の repo 内 dedup はどのノブでも止まらない**。
  したがって**既に断片化したファイルは現行クライアントでは回復不能** — 同じバイト列を同じ
  パスへ上げ直しても hf CLI が転送ごとスキップし、delete → 再 up の 2 コミット法でも不発
  （実測）。恒久対処の候補 3 案は [known-issues](known-issues.md) の text_encoder 断片化節。
