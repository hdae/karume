# 資産の置き場（models / outputs / inputs）

ローカル資産 3 根の規約。綴りの正本は `tools/export-recipes/_shared/paths.py`（`DIST_ROOT` /
`SERIES_ROOT` / `INPUTS_ROOT` / `OUTPUTS_ROOT`）。**3 根とも git 追跡しない**（全て再生成
可能な生成物か手置きの実重みで、リポジトリが持つのは作り方だけ）。

| 根                        | 中身                                                                 | 例                                                                           |
| ------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `models/`                 | **配布形だけ**（1 ディレクトリ = 1 HF リポ・そのまま上げられる）     | `models/karume-anima-turbo/` / `models/karume-sbv2-jvnv/`                    |
| `outputs/series/`         | exporter の系列出力（コンテナ + golden フィクスチャ `io.*`）         | `outputs/series/sbv2-F1-f16/`                                                |
| `outputs/series-archive/` | **裁定済みの系列の退避先**（消すと数時間の校正が消える — 下記）      | `outputs/series-archive/2026-08-23-anima-base-i4/`                           |
| `outputs/demo/`           | **デモ・ベンチの生成物**（`rm -rf outputs/demo` で常に安全に消せる） | `outputs/demo/*.png` / `outputs/demo/sbv2-dump/` / `outputs/demo/quant-sim/` |
| `outputs/`（その他）      | ホスト資産（消すと再取得・再エミットが要る）                         | `outputs/sbv2-demo/` / `outputs/yomi/`                                       |
| `inputs/<family>/<name>/` | 手置きの実重み（ckpt・config — 生成物ではない）                      | `inputs/sbv2/F1/`                                                            |

- `outputs/yomi/` の日本語辞書（`*.jtd`）の取得: HF dataset `hdae/yomi-dict` の
  `naist-jdic.jtd.gz` を解いて置く（無いと models の修正辞書テストは SKIP される）。
- `outputs/series-archive/<日付>-<件名>/` は、**配布しない裁定が出た系列を人手で退避する場所**。
  `series/` と違って `dist.py` からは再生成されず（校正に数時間かかる系列を「再生成可能」と
  誤読して掃除されるのを防ぐための別根）、`paths.py` にも定数を持たない — 退避も参照も手作業。
  復活レバーとしての位置づけは [backlog](backlog.md) later 節が持つ。

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
```

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
