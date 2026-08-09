# 資産の置き場（models / outputs / inputs）

ローカル資産 3 根の規約。綴りの正本は `tools/exporter/karume/paths.py`（`DIST_ROOT` /
`SERIES_ROOT` / `INPUTS_ROOT` / `OUTPUTS_ROOT`）。**3 根とも git 追跡しない**（全て再生成
可能な生成物か手置きの実重みで、リポジトリが持つのは作り方だけ）。

| 根                        | 中身                                                             | 例                                                                                   |
| ------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `models/`                 | **配布形だけ**（1 ディレクトリ = 1 HF リポ・そのまま上げられる） | `models/karume-anima-turbo/` / `models/karume-sbv2-jvnv/` / `models/karume-sbv2-fn/` |
| `outputs/series/`         | exporter の系列出力（コンテナ + golden フィクスチャ `io.*`）     | `outputs/series/sbv2-F1-f16/`                                                        |
| `outputs/`（その他）      | デモ・ベンチの出力とホスト資産                                   | `outputs/sbv2-demo/` / `outputs/yomi/`                                               |
| `inputs/<family>/<name>/` | 手置きの実重み（ckpt・config — 生成物ではない）                  | `inputs/sbv2/F1/`                                                                    |

## 組み立て（系列 → 配布形）

```sh
cd tools/exporter
uv run karume dist                                   # anima → models/karume-anima-turbo/
uv run karume dist --pipeline sbv2 --card-profile jvnv \
    --model F1 --model F2 --model M1 --model M2 --out ../../models/karume-sbv2-jvnv
```

- 仕様の正本は ADR [0041](decisions/0041-manifest-v2.md)（manifest v2・リポ内レイアウト =
  モデル別サブツリー + `shared/`・**配置は常に独立コピー** — ハードリンク禁止の理由も同 ADR
  追記）。`karume.json` は現物から導出（手書き禁止 — ADR 0038）。
- 組み立ては冪等（再実行で置き換え）。`verify_dist` が宣言と現物の突合・宣言外ファイル検査まで
  行い、モデルカード `README.md` は検証済み manifest から機械生成される（帰属は
  `--card-profile` — exporter の README 参照）。
- 系列を消しても配布形は壊れない（独立コピー）。逆に配布形は `karume dist` でいつでも系列から
  再生成できる。
