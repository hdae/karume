# 資産の置き場（models / outputs / inputs）

ローカル資産 3 根の規約。綴りの正本は `tools/export-recipes/_shared/paths.py`（`DIST_ROOT` /
`SERIES_ROOT` / `INPUTS_ROOT` / `OUTPUTS_ROOT`）。**3 根とも git 追跡しない**（全て再生成
可能な生成物か手置きの実重みで、リポジトリが持つのは作り方だけ）。

| 根                        | 中身                                                                 | 例                                                                                   |
| ------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `models/`                 | **配布形だけ**（1 ディレクトリ = 1 HF リポ・そのまま上げられる）     | `models/karume-anima-turbo/` / `models/karume-sbv2-jvnv/` / `models/karume-sbv2-fn/` |
| `outputs/series/`         | exporter の系列出力（コンテナ + golden フィクスチャ `io.*`）         | `outputs/series/sbv2-F1-f16/`                                                        |
| `outputs/demo/`           | **デモ・ベンチの生成物**（`rm -rf outputs/demo` で常に安全に消せる） | `outputs/demo/*.png` / `outputs/demo/sbv2-dump/` / `outputs/demo/quant-sim/`         |
| `outputs/`（その他）      | ホスト資産（消すと再取得・再エミットが要る）                         | `outputs/sbv2-demo/` / `outputs/yomi/`                                               |
| `inputs/<family>/<name>/` | 手置きの実重み（ckpt・config — 生成物ではない）                      | `inputs/sbv2/F1/`                                                                    |

- `outputs/yomi/` の日本語辞書（`*.jtd`）の取得: HF dataset `hdae/yomi-dict` の
  `naist-jdic.jtd.gz` を解いて置く（無いと models の修正辞書テストは SKIP される）。

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

- 仕様の正本は ADR [0041](decisions/0041-manifest-v2.md)（リポ内レイアウト = モデル別
  サブツリー + `shared/`・**配置は常に独立コピー** — ハードリンク禁止の理由も同 ADR 追記）と
  ADR [0071](decisions/0071-manifest-v3-shards.md)（format `karume/3` — shard 欄）。
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
export HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE=1000000
export HF_XET_DEDUPLICATION_MIN_N_CHUNKS_PER_RANGE_HYSTERESIS_FACTOR=1.0
export HF_XET_DEDUPLICATION_NRANGES_IN_STREAMING_FRAGMENTATION_ESTIMATOR=1
```

付けないと Xet の chunk 単位 dedup が効きすぎて**再構成が断片化し、DL が 5〜6 倍遅くなる**。
断片化は一度 CAS に載ると既定設定の再アップロードでは戻せない（片道ラチェット）ので、
**初回アップロードで防ぐのが唯一の低コストな手**である。機序と実測は
[research/2026-08-09-xet-fragmentation.md](research/2026-08-09-xet-fragmentation.md)。

- 上げたら**必ず検証する** — reconstruction の term 数を数え、`MiB/レンジ` が 10 を大きく
  下回っていないか見る（手順は同ドキュメント §9）。健全なら 1 xorb = 1 term に近くなる。
- 既に断片化してしまったファイルは、**同じバイト列のまま同じパスへ上げ直せば直る**
  （バイト列が変わらないので**コミットは増えない** — 変わるのは CAS の再構成記録だけ）。
  その際は `~/.cache/huggingface/xet/*/shard-cache` を退避してから実行する。断片化した祖先の
  shard がローカルに残っていると、そこへ dedup ヒットして元に戻る。
