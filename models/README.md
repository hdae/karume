# models/ — ローカル資産の索引（この README 以外は git 対象外）

再生成コマンドの正本は [tools/exporter/README.md](../tools/exporter/README.md)。E2E は資産が
無い系列を **SKIP** する（部分欠落は FAIL）ので、削除した系列のテストは再 emit まで走らない。

## 削除禁止（再入手の手段が無い）

| パス                                                                                      | 理由                             |
| ----------------------------------------------------------------------------------------- | -------------------------------- |
| `anima-turbo-lora-v0.2.safetensors`                                                       | 出所 URL の記録なし（143MB）     |
| `sbv2/jvnv-F1-jp_e160_s14000.safetensors` + `sbv2/config.json` + `sbv2/style_vectors.npy` | SBV2 元 ckpt — 配布元 URL 未特定 |

## 系列（全て再生成可能・`cd tools/exporter` して `uv run --group <g> python <script>`）

| ディレクトリ                           | 内容                                            | 使うテスト / デモ                     | 再生成                                                  |
| -------------------------------------- | ----------------------------------------------- | ------------------------------------- | ------------------------------------------------------- |
| `anima-f16/`                           | Anima 4 ターゲット f16                          | e2e_anima\*・デモ（テキスト経路/VAE） | `export_anima.py --dtype f16`                           |
| `anima-i8/`                            | DiT i8                                          | e2e_anima_i8                          | `export_anima.py --dtype i8`                            |
| `anima-f16-1024/`                      | DiT + VAE 1024px                                | e2e_anima_1024                        | 同上 `--resolution 1024`（README 参照）                 |
| `anima-turbo-f16/`                     | turbo DiT f16                                   | e2e_anima_turbo / f16compute          | `--target transformer --lora <LORA>`                    |
| `anima-turbo-i8/`                      | turbo DiT i8                                    | e2e_anima_w8a8                        | 同上 `--dtype i8`                                       |
| `anima-turbo-i8-1024/`                 | turbo DiT i8 1024px                             | e2e_anima_1024・e2e_anima_dyn         | 同上                                                    |
| `anima-turbo-f16-dyn/`                 | turbo DiT f16 **S 形**（解像度非依存）+ 素表    | e2e_anima_dyn・デモ `--dit-graph dyn` | 同上 + `--dit-graph dyn`                                |
| `anima-turbo-i8-dyn/`                  | turbo DiT i8 **S 形** + 素表                    | e2e_anima_dyn・デモ                   | 同上 `--dtype i8`                                       |
| `anima-pipeline*/`                     | 参照フィクスチャ（小）                          | 各 E2E                                | `anima_pipeline.py`（各系列の行を参照）                 |
| `anima-tiling-f16-1024/`               | VAE 固定タイル decode の参照（13MB）            | e2e_anima_tiling                      | `anima_tiling.py`（`anima-pipeline-turbo-f16-1024` 要） |
| `anima-rope-nonsquare/`                | 非正方 4 幾何の rope 表（16.5MB）               | e2e_anima_nonsquare                   | `anima_rope.py`（**重みを読まない** — 数秒）            |
| `anima-tiling-f16-1344x768/`           | 非正方タイル decode の参照（12MB）              | e2e_anima_nonsquare                   | `anima_tiling.py --resolution 1344x768 --latents …`     |
| `deberta/`                             | DeBERTa 2+24 層 f32                             | e2e_deberta・sbv2 デモ                | `export_deberta.py`                                     |
| `deberta-i8/`                          | 同 i8                                           | e2e_deberta / w8a8・sbv2 デモ         | `export_deberta.py --dtype i8`                          |
| `sbv2/`                                | SBV2 5 ターゲット f32 + **元 ckpt（削除禁止）** | e2e_sbv2・relattn parity・デモ        | `export_sbv2.py`                                        |
| `sbv2-f16/` `sbv2-i8/`                 | 同 f16 / i8 系列                                | e2e_sbv2・デモ                        | `export_sbv2.py --dtype {f16,i8}`                       |
| `sbv2-demo/` `anima-demo/text` `yomi/` | デモ実行時資産（トークナイザ表・辞書・dump）    | デモ・tokenizer テスト                | `sbv2_demo.py assets` / `anima_demo.py` / 辞書は初回 DL |

## 意図的に置いていない系列（削除済み — 必要なら再 emit）

- `anima/`（f32 4 ターゲット・11GB）: e2e_anima_test.ts の f32 系列が SKIP になる（DiT 2 本は
  元々 GPUBuffer 天井で ignored — docs/limitations.md）。再生成 `export_anima.py`（約 15 分）。
- `anima-turbo-f16-1024/`（3.7GB）: デモ専用（`--turbo --dit f16 --resolution 1024`）。
  テスト参照なし。

## 生成物の置き場

- `anima-demo/out/` — **PNG sha256 門の基準 2 枚は消さない**（`turbo-f16-512-seed42.png`
  `20990ae4…` / `turbo-i8-1024-seed42.png` `ce6c950f…` + 同名 .json）。他は使い捨て。
- `sbv2-demo/out/` — `dump.safetensors` / `reference.wav`（quant-sim の恒真化検査の突合先）/
  `out.wav`（パリティ対）/ `official.wav` は残す。
- `sbv2-demo/quant-sim/` — 量子化聴感ゲートの WAV（`measure_quant_sbv2.py` で約 20 秒で再生成）。
- `anima-demo/q0-w8a8/` — ADR 0025 の目視裁定画像（記録物 — 消さない）。
