# Karume

WebGPU で動く汎用 NN 推論スタック（Deno + ブラウザ両対応・純 TypeScript + WGSL・ランタイム依存
ゼロ）。JSR の 3 パッケージ — `@karume/runtime`（IR 実行）/ `@karume/hub`（モデルの解決・取得・
キャッシュ）/ `@karume/models`（パイプラインと tokenizer）— と、PyTorch のモデルを IR へ落とす
PyPI パッケージ `karume`（エクスポータ CLI）で構成する。

Status: **WIP**（本記述は公開準備の段で書く。現在の設計は [docs/decisions/](docs/decisions/) が正本）

License: MIT（[LICENSE](LICENSE)）
