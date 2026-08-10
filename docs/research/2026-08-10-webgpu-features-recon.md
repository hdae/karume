# WebGPU features の活用度と余地（subgroups / DP4a / immediates / subgroup matrix）

> 時点スナップショット（2026-08-10）。実機 = RTX 3080 Ti / Linux / Deno 2.9.4（wgpu 29.0.1）。
> 外部状況の出典 URL は取得日 2026-08-10。裁定の文脈: prepared 機構（ADR 0042）の維持判断と
> LLM 対応の見通しと合わせて評価した。

## 1. 実機プローブ（Deno 2.9.4・requestAdapter 列挙のみ）

- `adapter.features`: **immediates あり**・shader-f16・timestamp-query ほか。**subgroups 無し**。
- `wgslLanguageFeatures`: packed_4x8_integer_dot_product（= DP4a 有効）・pointer_composite_access・
  readonly_and_readwrite_storage_textures。
- `maxImmediateSize`: **undefined**（limit 未露出）。`adapterInfo`: 空 serialize だが
  `subgroupMinSize/MaxSize = 32/32`（ハードは subgroup 32 レーンを報告している）。

## 2. 現状の活用度（コード実査）

- 要求 feature の既定は **0 本**（REQUIRED_FEATURES = []・device.ts:133）。opt-in 2 本 =
  timestamp-query（gpuTiming・ADR 0021/0032）・shader-f16（要求 + 実走カナリア突合 —
  device.ts:236-316）。limits 11 本はアダプタ実測値をそのまま要求し、granted 不足は
  GpuLimitError（黙って能力を落とした device を返さない — device.ts:110-120）。
- **DP4a は使用済みかつ i8a8 経路で完全**: 整数内積の実体は linear-i8a8.ts の idot 1 本で、
  linear / attention QK / PV の 3 カーネルが共有。選択は `wgslLanguageFeatures` の列挙で
  Session 構築時に 1 度（既定自動・executor.ts:880）。dp4a/emu は**数値完全一致**（i32 厳密
  加算）なので誤選択でもビットは動かない。唯一の沈黙縮退 = wgslLanguageFeatures 未提供
  環境で emu に落ちる（観測点はパイプラインキー `:dp4aEmu` のみ）。
- subgroups / subgroup matrix / immediates の要求・使用は**ゼロ**（融合カーネルは
  「optional feature 非依存」を設計方針として明文化 — ADR 0040:134）。

## 3. 外部状況（2026-08 時点・一次ソース優先）

| feature                                | spec                                                                                  | Chrome                                       | wgpu                                                   | **Deno**                                                                                                                                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| subgroups                              | proposal → 出荷済み                                                                   | **134 で stable**                            | SUBGROUP（native flag）                                | **列挙されず要求不可**（webidl は FromStr 素通しだが adapter が広告しない）                                                                                                                     |
| subgroups-f16                          | 廃止方向                                                                              | shader-f16 + subgroups の併用へ              | —                                                      | —                                                                                                                                                                                               |
| DP4a（packed_4x8_integer_dot_product） | WGSL 言語拡張（feature 名ではない）                                                   | 123〜                                        | v26〜列挙対応                                          | **有効（実測）**                                                                                                                                                                                |
| immediates                             | **本体 spec 入り 2026-05-07**（gpuweb#5423・maxImmediateSize 制御・var\<immediate\>） | **対応済み（149-150 期・ユーザー確認済み）** | v28 で rename 実装・v29 系で精緻化                     | **adapter.features に列挙されるが API 未露出**（device.rs が `immediate_size: 0` 固定・setImmediateData 無し）→ wgpu 側は対応済みで **Deno の露出待ち**（ユーザー見立て「そのうち来る」と整合） |
| subgroup matrix                        | 提案段階（gpuweb#4195・WGSL 会議進行中）                                              | pre-origin-trial（設計調査段階）             | EXPERIMENTAL_COOPERATIVE_MATRIX（v29.0.4 で MSL 対応） | 露出なし                                                                                                                                                                                        |

主要出典: gpuweb/gpuweb#5423（immediates spec merge）・proposals/immediate-data.md（feature 名
無し・maxImmediateSize 既定 64B）・denoland/deno ext/webgpu/{webidl,device,pipeline_layout}.rs
（実装読み）・gfx-rs/wgpu CHANGELOG（v26/v28/v29 各節）・developer.chrome.com new-in-webgpu
134/145/149-150（二次）。

## 4. カーネル適用余地の写像（コード実査の帰結）

- **subgroups**: 行縮約族 5 本は共通骨格（1 行 = 1 WG 256・共有 1KB・barrier 10 本/パス）。
  - f32 の**総和**縮約（layer_norm / rms / softmax② / stats②）への適用は**ビット同一門と
    正面衝突**（縮約順が変わる — attention.ts:21-28 が明文で禁止・変えるなら別キー +
    tolerance 全面再導出 = ADR 0022 の規律）。
  - 門と衝突しない適用面 = **max 縮約のみ**（softmax① / attention_stats① — 有限値で順序
    非依存）。ただし quantize_rows の amax は **NaN ビット伝播の MUST（ADR 0020）**があり
    組み込み subgroupMax 不可。
  - i8a8 GEMM 族には**レーン間縮約が存在しない**（per-thread レジスタ acc）— 適用余地ゼロ。
  - 触れる GPU 時間の上限 ≈ DiT の 14%（stats 5.46% が単独最大）。**本機 Deno で実走不能**な
    ため現時点では実装しても門を掛けられない（ADR 0005 と衝突）。→ **Deno 露出待ち・
    露出後に stats① から**。
- **subgroup matrix**: f32 は ADR 0022 と正面衝突。唯一門を割らない面は i8a8（順序非依存）
  だが per-thread 縮約からの骨格再設計になる。標準化 pre-OT・Deno 露出なし —
  **ウォッチ対象**（テンソルコア級の利得はここにしかない・LLM 文脈で将来重要）。
- **immediates**: 数値ビット不変（門の再実測不要・fixture は全面更新）だが、**波 1+2 着地後の
  現行機構では利得が薄い、むしろ逆行**: backed run は params 焼き込み済みで run 中の操作は
  dispatch のみ — immediates 化は毎 dispatch の setImmediateData を**足す**方向。params
  確保・転送は HOST-006 の内容アドレスキャッシュで Session 生涯 1 度に既に落ちている。
  加えて layout:"auto" の放棄（30 本超の明示 layout 化）・binding 0 撤去の全 WGSL 改番・
  lastRunParams 診断の意味喪失が主コスト。→ **採用見送り（Deno 露出後も急がない）**。
  維持理由が生じるのは「bind group 本数自体を削りたい」将来の別設計時。
- **DP4a の使い残し**: i8a8 経路には無し。ただし **i8 格納 conv（VAE conv2d = GPU 19.1%）は
  重み逐次 dequant の f32 MAC で dp4a 恩恵ゼロ**（ADR 0019 の数値契約の帰結・実装漏れでは
  ない）。効かせるには活性側も i8 化した **i8a8 conv 新経路**が要る — feature 待ちではなく
  設計仕事で、次のカーネル波の最有力候補。

## 5. 帰結（prepared 機構・staged execution との合流）

- 新 feature はいずれも「今すぐ本機で使えて門と整合するもの」が無い（immediates は使えるが
  波 1+2 が同じ問題を既に解いた後・subgroups/coop matrix は Deno 露出待ち）。
- prepared 機構（ADR 0042）は feature 進化と独立に成立しており、subgroups/coop matrix が
  来た場合もレシピ層はカーネル選択の下流なので**干渉しない**。immediates が来ても採用しない
  判断（上記）なので衝突しない。
- 次の実利は feature ではなく設計仕事側にある: **i8a8 conv**（19.1% への dp4a 適用）・
  カーネル本線（DiT linear+attention 63.3%）・ロード時間。
