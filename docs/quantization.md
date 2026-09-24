# 量子化方式の索引（quantization）

> 正本は各セルの参照先。ここは索引 — 数値と意味論を複製せず、どこを読めば決まるかだけを持つ。
> 用語の定義は [glossary](glossary.md)、性能候補の採否は [perf-ledger](perf-ledger.md)、
> by-design 制約は [limitations](limitations.md)、波順は [backlog](backlog.md) が正本。
> 状態語彙: `製品既定` / `任意`（席や実行ノブとして選べる）/ `実験`（family 限定）/
> `宣言のみ`（受理するが実行経路が無い）/ `保留`（測定のみ・格納経路なし）/ `棄却`（記録として保持）。

## 表 1 — 格納型（資産に焼く形）

正本 = codec 台帳（`packages/runtime/src/format/container/codecs.ts` の `CODEC_LEDGER`・
[container-v1](container-v1.md) §6.3）と、意味論 dtype との対応を持つ合流層の `allowedLayouts`
（`packages/runtime/src/format/container/bind.ts`・[ir-v2](ir-v2.md) の「値と型」節）。
表の格納型は codec の展開経路の名前で、登録名との対応は `int8-sym` → `i8`・`int4-sym-g` → `i4`・
`int2-off` / `ternary` → `i2`。ADR 0006 以来「意味論は f32、格納だけを圧縮する」が骨格で、
`i32` だけが例外（記号依存定数の生の int32）。

| 格納型                             | 使うファミリ                                 | 状態                                                                       | 正本                                                                                                                                                                                                |
| ---------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `f32`                              | 全ファミリ                                   | 製品既定                                                                   | ADR [0006](decisions/0006-quantization.md)                                                                                                                                                          |
| `f16`                              | anima / sbv2 / irodori                       | 任意                                                                       | ADR [0018](decisions/0018-f16-weight-execution.md)                                                                                                                                                  |
| `i8`                               | sbv2 / irodori / gemma4 の埋め込みと drafter | 製品                                                                       | ADR [0019](decisions/0019-i8-weight-execution.md)                                                                                                                                                   |
| `i4`                               | anima / sbv2 / irodori / gemma4 / gemma4-qat | 製品                                                                       | ADR [0069](decisions/0069-packed-w4-storage.md)                                                                                                                                                     |
| `i2`                               | gemma4-qat のみ                              | 実験（family 限定・linear と embedding だけ・`linearCompute: "f32"` 限定） | ADR [0097 追記 1](decisions/0097-gemma4-qat-integration.md#追記-1--int2-格納と実行の契約2026-09-11)                                                                                                 |
| `bf16`                             | 無し                                         | 宣言のみ（実行は fail loudly）                                             | ADR [0006](decisions/0006-quantization.md)                                                                                                                                                          |
| `i32`                              | 記号を持つ全ファミリ                         | 製品（記号依存定数の焼き込み先）                                           | ADR [0010](decisions/0010-symbolic-constant-folding.md)                                                                                                                                             |
| PLE（IR の外・`model` 容器の資産） | gemma4 / gemma4-qat                          | 製品（索引 schema 3・`storage` = `i8`〈省略時〉/ `i4` / `i2`）             | ADR [0085](decisions/0085-ple-host-gather.md)・[0109](decisions/0109-manifest-v5-container.md) 決定 4・[0097 追記 4](decisions/0097-gemma4-qat-integration.md#追記-4--packed-ple-sidecar2026-09-11) |

## 表 2 — 実行ノブ（`SessionOptions`）

正本 = `packages/runtime/src/runtime/session-types.ts` の `SessionOptions`。
manifest へ保存できるキーの正本は `packages/hub/src/manifest.ts` の `SESSION_KEYS`、
数値を変えるノブの opt-in 契約は ADR [0058](decisions/0058-numerics-opt-in-contract.md)。

| ノブ                         | manifest 保存                                | 状態                                                    | 正本                                                                                                                                                                             |
| ---------------------------- | -------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `linearCompute`              | 可                                           | 任意（`a8` は w8a8 / w4a8・`f16` は `shader-f16` 必須） | ADR [0025](decisions/0025-w8a8-linear-execution.md)・[0028](decisions/0028-f16-compute-variants.md)・[0076](decisions/0076-w4a8-linear-execution.md)                             |
| `attentionCompute`           | 可                                           | 任意                                                    | ADR [0028](decisions/0028-f16-compute-variants.md)・[0030](decisions/0030-attention-a8-execution.md)                                                                             |
| `attentionScoreStorage`      | 可                                           | 任意                                                    | ADR [0031](decisions/0031-attention-score-f16-storage.md)                                                                                                                        |
| `linearGemvReduce`           | 一部可（`parallel-subgroup32` は保存語彙外） | 任意（`i4-gemvpar` / `i4-fast` が宣言）                 | ADR [0098](decisions/0098-linear-gemv-parallel.md)・[0101](decisions/0101-linear-gemv-subgroup.md)                                                                               |
| `fuseRmsNormAdd`             | 可                                           | 任意（`i4-fast` が宣言）                                | ADR [0099](decisions/0099-rms-norm-add-fusion.md)                                                                                                                                |
| `fuseLinearStaticQuantize`   | 可                                           | 任意（gemma4-qat の `i4-fast` だけが宣言）              | ADR [0103](decisions/0103-linear-static-quantize-fusion.md)・[0104](decisions/0104-gemma-fast-quant.md)                                                                          |
| `rmsNormReduce`              | 不可                                         | 任意（subgroups 必須）                                  | ADR [0100](decisions/0100-rms-subgroup-reduction.md)                                                                                                                             |
| `stateAttentionReduce`       | 不可                                         | 任意（Gemma family 既定は `parallel`）                  | ADR [0058](decisions/0058-numerics-opt-in-contract.md)・[0067](decisions/0067-autoregressive-attention-vocabulary.md)・[0102](decisions/0102-state-attention-stats-pv-fusion.md) |
| `linearGemvRowsThreadTarget` | 不可                                         | 任意（並列度の目標値）                                  | ADR [0022](decisions/0022-gemm-register-blocking.md)・[0082](decisions/0082-linear-gemv-decode.md)                                                                               |
| `planBackingBudgetBytes`     | 不可                                         | 任意（保持予算）                                        | ADR [0095](decisions/0095-plan-backing-budget.md)                                                                                                                                |
| `submitPolicy`               | 不可（ADR 0104 が明示的に除外）              | 任意（ホスト政策）                                      | ADR [0004](decisions/0004-execution-model.md)                                                                                                                                    |
| `sharedWeights`              | 不可                                         | 任意（借用の門 5 点）                                   | ADR [0096](decisions/0096-speculative-decoding.md)                                                                                                                               |
| `gpuFeatures.shaderF16`      | 可（`session` の外・`Quant.gpuFeatures`）    | 任意                                                    | ADR [0038](decisions/0038-manifest-v1.md)                                                                                                                                        |
| `requiredLimits`             | 可（`Quant.requiredLimits`）                 | 任意（重み取得前の admission で拒否）                   | ADR [0038](decisions/0038-manifest-v1.md)・[0089](decisions/0089-memory-limits-preflight.md)                                                                                     |

## 表 3 — quant 席（配布上の名前）

正本 = 各 `tools/export-recipes/<family>/distribution.py` の quant 表。
命名規則は ADR [0074](decisions/0074-quant-seat-naming.md)、表示名と説明は [0075](decisions/0075-quant-presentation.md)。
★ = その model の `defaultQuant`。

| ファミリ                                             | 席                                                                                              | 状態                                                                    | 正本                                                                         |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| anima                                                | `f16` / `f16+dit8` / `f16+dit8-a8` / `f16+dit8-a8-attn8` / `f16+dit8-a8-attn8-s16`★ / `f16-c16` | 製品                                                                    | `anima/distribution.py`                                                      |
| anima                                                | `f16+dit4` / `f16+dit4-attn8-s16`                                                               | 任意（配布スキップ裁定 2026-08-24 — [perf-ledger](perf-ledger.md) Q-5） | `anima/distribution.py`                                                      |
| sbv2                                                 | `f16+bert8` / `i8` / `i8-a8` / `i8+bert4`★ / `i4`                                               | 製品                                                                    | `sbv2/distribution.py`                                                       |
| irodori                                              | `f32` / `f16` / `i8` / `i8-a8`★ / `i8+dit4`                                                     | 製品                                                                    | `irodori/distribution.py`                                                    |
| birefnet / depth_anything / siglip2 / vowel_detector | `f32`★                                                                                          | 製品                                                                    | 各 `distribution.py`                                                         |
| gemma4 e2b                                           | `i4` / `i4-gemvpar` / `i4-fast`★                                                                | 製品                                                                    | `gemma4/distribution.py`・ADR [0104](decisions/0104-gemma-fast-quant.md)     |
| gemma4-qat e2b                                       | `i4` / `i4-gemvpar` / `i4-fast`★                                                                | 実験（family 全体が実験段階 — [limitations](limitations.md) の QAT 節） | `gemma4_qat/distribution.py`・ADR [0104](decisions/0104-gemma-fast-quant.md) |
| gemma4-qat e4b                                       | `i4`★                                                                                           | 実験                                                                    | `gemma4_qat/distribution.py`                                                 |

## 表 4 — exporter 側の量子化方式

### 4-a. 自動量子化（重みを丸めてから焼く）

正本 = `tools/exporter/src/karume/quantize.py` / `quant_calib.py` / `quant_methods.py`。
`quant_methods.py` / `quant_calib.py` のモジュール docstring が「emit へ渡す口を作らない」を MUST として
宣言しており、**格納経路を持つのは f16 / RTN i8 / RTN i4 / GPTQ-rtn i4 の 4 つだけ**という線引きを
コードが担保する。

| 方式                                | 格納経路                 | 状態             | 正本                                                                                                          |
| ----------------------------------- | ------------------------ | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| f16 丸め                            | あり                     | 製品             | ADR [0018](decisions/0018-f16-weight-execution.md)                                                            |
| RTN i8                              | あり                     | 製品             | ADR [0019](decisions/0019-i8-weight-execution.md)・[0029](decisions/0029-sbv2-i8-series-and-quant-quality.md) |
| RTN i4                              | あり                     | 製品             | ADR [0069](decisions/0069-packed-w4-storage.md)                                                               |
| GPTQ（`grid="rtn"`）                | あり（i4 席へ直結）      | 製品             | [perf-ledger](perf-ledger.md) Q-6                                                                             |
| GPTQ の act-order / static-groups   | あり（格納形は不変）     | 任意・既定オフ   | [backlog](backlog.md)「GPTQ 掃引の再評価」                                                                    |
| AWQ                                 | 無し                     | 棄却             | [perf-ledger](perf-ledger.md) Q-7                                                                             |
| NF4                                 | 無し（測定専用）         | 保留             | [perf-ledger](perf-ledger.md) Q-3                                                                             |
| FP4（e2m1）                         | 無し                     | 保留（測定のみ） | `quant_methods.py`                                                                                            |
| MXFP4                               | 無し                     | 棄却             | [perf-ledger](perf-ledger.md) Q-4                                                                             |
| k-means codebook                    | 無し                     | 保留             | [perf-ledger](perf-ledger.md) Q-2                                                                             |
| 群内直交回転（rot-rtn / rot-lloyd） | 無し（未実装）           | 保留（起票のみ） | [perf-ledger](perf-ledger.md) Q-11                                                                            |
| 活性の per-token i8 化              | 格納ではなく参照側の模擬 | 製品             | ADR [0025](decisions/0025-w8a8-linear-execution.md)・[0030](decisions/0030-attention-a8-execution.md)         |

### 4-b. 固定量子化（QAT — 上流の整数をそのまま持ち込む）

| 項目                                                                                                              | 状態                                                                                                                                                                                                                                                                                                                  | 正本                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 入口 `FixedQuantizedWeight` + `publish_model(fixed_weights=...)`（受理 dtype `i2` / `i4` / `i8`・混成可）         | 製品                                                                                                                                                                                                                                                                                                                  | ADR [0097 追記 3](decisions/0097-gemma4-qat-integration.md#追記-3--固定量子化-writer-の入口2026-09-11)         |
| 供給元 `gemma4_qat/checkpoint.py`（上流の packed バイト列と scale を退避し、trace には shape だけのダミーを挿す） | 実験                                                                                                                                                                                                                                                                                                                  | ADR [0097 追記 5](decisions/0097-gemma4-qat-integration.md#追記-5--固定-qat-recipe-と数値比較の扱い2026-09-11) |
| 活性側 SRQ（`karume::static_quantize` → 同名 IR op）                                                              | 実験                                                                                                                                                                                                                                                                                                                  | ADR [0097 追記 2](decisions/0097-gemma4-qat-integration.md#追記-2--固定-srq-op-の契約2026-09-11)               |
| packed PLE（索引 schema 3 の `storage` = `i2` / `i4`）                                                            | 実験                                                                                                                                                                                                                                                                                                                  | ADR [0097 追記 4](decisions/0097-gemma4-qat-integration.md#追記-4--packed-ple-sidecar2026-09-11)               |
| 活性の整数内積（公式 mobile と同じ計算形）                                                                        | 段 1a 採用（QAT E2B の `i4-fast` が `packedStaticQuantize: true` を宣言・明示 false で外せる・[ADR 0105 追記 2](decisions/0105-packed-static-quantize-activations.md)・Chrome +8.3%・[research §16](research/2026-09-19-qat-speed-recon.md)）・段 1b（整数内積）は棄却 — lm_head に int8 活性が無く上限 0.09 ms/token | [perf-ledger](perf-ledger.md) K-45                                                                             |
| KV cache の int8 化（公式 `k/v_cache_scale`）                                                                     | 未実装・速度には効かない（[2026-09-19](research/2026-09-19-qat-speed-recon.md) §7.2・メモリ項目として扱う）                                                                                                                                                                                                           | [perf-ledger](perf-ledger.md) K-46                                                                             |
| 自動 INT2 量子化（`publish_model(weight_dtype="i2")`）                                                            | 持たない（固定経路だけ）                                                                                                                                                                                                                                                                                              | ADR [0097 追記 1](decisions/0097-gemma4-qat-integration.md#追記-1--int2-格納と実行の契約2026-09-11)            |
