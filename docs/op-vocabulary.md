# op 語彙台帳 — Core ATen 160（プロトタイプからの引き継ぎ）

出自: 先行実験プロジェクト（以下プロトタイプ）の `docs/op-vocabulary.md`（2026-08-01 時点）
の引き継ぎ。ユーザー指示によりプロトタイプで進行中だった「必要な OP の検討」を
Karume が継続する。以後の正本は本ファイル。

NOTE: 25/15 の個別列挙は本書に取り込み済み。75/38/7 の箱の per-op 割当はどこにも存在せず、
M1-P2 以降のエクスポート実測（未対応 op の全件列挙）で確定していく。

## 母集団の確定（引き継ぐ結論）

- 母集団は **Core ATen IR の 160 op**（`torch.Tag.core`）。aten overload 2879・関数名 1302・
  decomp テーブル 943 に対し、torch.export の `run_decompositions` がここまで潰す。**ここが
  上限**であって TODO リストではない。160 を埋めること自体は目標にしない。
- 逆に潰されすぎると困る 12 op（下記「分解禁止 12 op」）はエクスポータ側で分解を止めている —
  融合カーネル 1 本の方が速いから。この非対称は正しい状態で、プロトタイプの
  `attention` / `rms_norm` / `silu` が 160 に含まれないのも同じ理由（`attention` は
  ADR 0023 で Karume 側の語彙に入った — 保存対象はターゲット別）。

## 5 分類（160 の全割当）

| 分類             | 数 | 内容                                              |
| ---------------- | -- | ------------------------------------------------- |
| 実装しなくてよい | 25 | backward / 乱数（ホスト側責務）/ 形状メタ / 3D 系 |
| 必須プリミティブ | 75 | プロトタイプ実装済 ≈45。残りが実装作業の本体      |
| 組み合わせで済む | 38 | 新カーネル 0。既存 op への分解・正規化で吸収      |
| あったら嬉しい   | 7  | 優先度低                                          |
| 難しい・遅い     | 15 | 下記 3 系統（個別 ADR 前提）                      |

実装対象は 120（= 75 + 38 + 7）。

### 実装しなくてよい（25）の内訳

- **学習専用（7）**: `*_backward` 系。forward-only export には出ない。
- **乱数（4）**: `rand` / `randn` / `randperm` / `native_dropout`。diffusion の初期ノイズは
  **ホスト生成が正しい**（呼び出し側アプリの責務であり、推論ランタイムの責務外）。
- **形状メタ・別名（9）**: `sym_size` / `sym_numel` / `sym_stride` / `sym_storage_offset` は
  エクスポータが SymInt 式を評価して記号次元（`T`, `8T+2` 等）へ畳むためノードを出さない。
  `alias` はプロトタイプ側のスキップ対象語彙で処理済み。`_local_scalar_dense` /
  `as_strided` / `empty` / `empty_strided` は受理集合外 — 出たら fail loudly が正しい
  （黙って近似しない、という横断の不変条件どおり）。
- **3D・動画専用（5）**: 2D 版が固まってから同型で足せる。

## 難しい・遅い（15）の 3 系統 — 個別列挙

1. **散布（scatter_add / index_put / scatter / scatter_reduce / masked_scatter / col2im）** —
   WGSL に `atomic<f32>` が無く、CAS 実装は加算順が非決定になり codegen/実行の決定性
   （同一入力 → 同一出力）を静かに壊す。添字が export 時に静的なら **gather へ反転**
   （逆写像を定数化）する方が噛み合う。
2. **データ依存出力形状（nonzero / bool マスク index / _embedding_bag）** — executor は
   全ノードの出力 shape を実行前に確定させる静的形状前提を持ち、記号次元は入力の要素数
   からしか束縛されない。**GPU 結果を読み戻して形状を決める経路が存在しない**。対応する
   なら「固定上限形状 + マスク」か、実行モデル自体の設計裁定が要る。
3. **値依存の並べ替え・その他（sort / topk / grid_sampler_2d / _cdist_forward /
   _pdist_forward / _fft_r2c / _fft_c2r）** — 保留。`topk` は k=1 なら `argmax` で代替できる。

NOTE: 個別列挙の合計は 16（6+3+7）で箱の計 15 と 1 件合わない。この不整合はプロトタイプ
原本由来（どの op が二重計上かは原本にも記録が無い）。per-op 割当を実測で確定する
M1-P2 以降に解消する。

## 引き継ぐ設計知見

- **最大の穴 = 値の max reduce 族**: amax / amin / max / min / argmax / argmin が皆無
  だった。これが無いと safe-softmax・max-pool・logsumexp・greedy デコードが組めず、
  Transformers 系を汎用に扱えない。実装は sum 用の行カーネルと同型の複製 ≈20 行で済む
  ため最初期に実装する。
- **分解禁止（融合維持）12 op**: linear, layer_norm, rms_norm, softmax, gelu, leaky_relu,
  conv1d, conv2d, conv_transpose1d, embedding, masked_fill, **scaled_dot_product_attention**
  （`leaky_relu` は ADR 0015 で 9 → 10。分解形 `gt_scalar + mul + where` は中間バッファが
  1.5〜2 倍に膨らむ。`rms_norm` は ADR 0017 で 10 → 11 — こちらは**保存だけでは足りず**、
  手書き分解形を拾う畳み込みパスと 2 系統で供給する。`scaled_dot_product_attention` は
  ADR 0023 で 11 → 12）。
  **12 本目だけは既定の保存リストに載らない**（`PRESERVED_OP_PREFIXES` は 11 本のまま）—
  SDPA は mask / causal / GQA を引数で表せてしまい、グローバルに保存すると契約外の形
  （kwargs 渡しの bool mask・`[B,1,M,N]` 等）が `_h_attention` の fail loudly に当たって
  export できなくなる。加算型 f32 `[1,1,M,N]` の mask だけは 2026-08-11 の改訂（ADR 0023
  追記）で語彙に入った。保存は `curated_decompositions(preserved=…)` を通した
  **ターゲット別の opt-in**（`PRESERVED_OP_PREFIXES_WITH_ATTENTION` — Anima の transformer /
  vae_decoder と EmbeddingGemma）。
  エクスポータ側で分解を止める — 融合カーネル
  1 本の方が分解後の複数 dispatch より速いという判断で、この非対称は正しい状態。
  プロトタイプの attention / rms_norm / silu が Core ATen 160 に含まれないのも同根。
  分解禁止リストは速度に直結し、後付けはグラフ層の再設計になる。
- **「分解できる」の反例集**（プロトタイプの敵対検証で判明。分解案は torch 突合ゲート必須）:

  | 主張                                       | 判定 | 理由                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
  | ------------------------------------------ | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `log` = `log1p(x−1)`                       | 誤り | `log1p` は WGSL 組込みでなく `log(1+a)` を生成するので、`x−1` の丸めがそのまま戻る。x=1e-6 で約 6% ずれる。単項 log 用の codegen 分岐は既にあり executor に 1 枝足すだけ                                                                                                                                                                                                                                                                                                                                                                                                                            |
  | `_log_softmax` = `log(softmax(x))`         | 誤り | 上の誤差を y≪1 の領域で踏む。underflow した要素が `log(0)=−inf` になり torch の有限値と不一致。**op の存在理由を消している**                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  | `pow` は WGSL 組込みで 1 行                | 誤り | WGSL の `pow` は**負の底が未定義**。`torch.pow(-2,3) = -8` と不一致。整数指数は繰り返し `mul` へ（実装済み）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  | `maximum`/`amax` は内部の `max` を公開     | 誤り | 「WGSL の `max` は NaN 伝播を保証しない」ため `leaky_relu` は select 形（ADR 0015・M1-P3 波3 で実装）。**実測（2026-08-02）**: select 形でもコンパイラが max イディオムへ畳み、ドライバの `max` が NaN を飲む（`clamp(NaN,-1,1) = -1` / `clamp_min(NaN,0) = 0` / `relu(NaN) = 0`。比較単体は仕様どおり false になる）。**根治済み（2026-08-03）**: `clamp` / `clamp_min` / `relu` / `amax` / `amin` は**ビット列 NaN 判定**（`(bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u`）の外殻で伝播させる — 整数比較なので畳み込みの対象にならない。`leaky_relu` だけは両枝に x が現れるため select 形のまま |
  | `mean`/`var`/`group_norm` = 素の `sum`     | 誤り | sum のリファレンス実装は 1 スレッド=1 行の逐次走査。group_norm は行数 32・行長数百万という真逆の形状で、32 スレッドが数百万回ループする。**2 段縮約なら成立**                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  | `var` = `sum((x−mean)²)/N`                 | 誤り | `torch.var` の既定は `correction=1`（N−1）。N 割りは norm 系の内部分散だけ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  | `group_norm` → `layer_norm` へ委譲         | 一部 | 正規化は合うが **affine が合わない** — layer_norm の weight 長は最終次元前提のため、解像度依存の巨大 ones/zeros が要る。別途 `[1,C,1,1]` broadcast の 2 パスが必要                                                                                                                                                                                                                                                                                                                                                                                                                                  |
  | tensor 比較 = `sub` + `*_scalar` 比較      | 誤り | `a=b=+inf` で `a−b=NaN` となり `eq` が false（torch は true）。attention の −inf ガードで現実に出る                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  | `floor`/`ceil`/`round`/`trunc` = cast 往復 | 誤り | WGSL 組込みがあり **1 行の codegen 追加**。cast 経路は `\|x\| ≥ 2^31` で壊れる。丸め規約は ties-to-even で torch と一致                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
  | `fmod` は手書きが要る                      | 誤り | WGSL の `%` は `e1 − e2·trunc(e1/e2)` = `torch.fmod`。**`remainder` の方**が floor 除算で符号補正が要る。整数型は両方とも封じられている                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
  | `expm1` = `exp(x)−1` は厳密                | 誤り | x≲1e-8 で `exp(x)` が 1.0 に丸まり結果 0。torch は x を返すので相対誤差 100%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  | 非 1 軸 `repeat` = unsqueeze+expand+view   | 誤り | rank4 入力で中間が rank5 になり strided 表現の rank 上限（4）を超える。使えるのは rank≤3 まで                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  | `select_scatter`/`slice_scatter` = `cat`   | 未確 | 境界（index 0 / 末尾）で長さ 0 のスライスが出て、WebGPU の storage binding が size>0 を要求するため落ちうる（未実測）。空オペランドを畳む分岐が必須                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  | `batch_norm` の 6 ステップ展開             | 成立 | 数値は同値。ただし elementwise 融合が無い前提だと 1 ノード = 1 dispatch + フルサイズ中間 1 本になる。1024px VAE の `[1,128,1024,1024]` = 537MB では容量が問題になる                                                                                                                                                                                                                                                                                                                                                                                                                                 |

  NOTE（2026-08-11）: 表の「tensor 比較」行が指す **attention の −inf ガード**は、ガード部分木を
  op 化する案（`eq(−inf)` 等を語彙へ）を却下したまま **`safe_softmax` 1 op** で決着した — ADR
  [0044](decisions/0044-runtime-attention-mask.md)。ガードの**意味論**（全 −inf 行 → 0）だけを
  `softmax` の変種として持つので、非有限値を扱う比較 op は語彙に入らない。

## やらない方がよいこと

- **汎用 reduce フレームワークを先に作らない。** amax は sum 用行カーネルの複製約 20 行が
  正解。抽象化の元になる実装がまだ 2 つ（sum / cumsum）しかない段階で先に一般化すると、
  形状が定まる前の抽象化になり手戻りが大きい。
- **torch の decomposition テーブルを移植しない。** 943 エントリあるが、実際に書く分解は
  38 op ぶんで大半が数行。プロトタイプの前例（select / split_with_sizes / pow(2) 相当）と
  同じ粒度で個別に足す。
- **Core ATen に無い op を「対称性のため」足さない。**
- **160 全部を埋めることを目標にしない。**
- **`scatter_add` を `atomic<f32>` の CAS ループで安易に入れない** — 加算順が非決定になり
  「同一入力 → 同一出力」を静かに壊す。

## op 追加の判定手順（第 0〜3 層）

DECIDED: [0043](decisions/0043-op-addition-layers.md)（判定軸の一本化と第 1' 層は同 ADR の
2026-08-12 追記）。未対応 op が `UnsupportedAtenOpsError` に出たら、上から順に**安い層**へ
落とす。実行の実体は全層カーネル。**第 1/2 層を分ける軸は原子/分子**（語彙内の他 op の
合成で厳密同値に書けるか）、**第 2/3 層を分ける軸は「分子の名前がどこに書かれるか」**
（IR ファイル = 第 2 層 / ランタイム内部のみ = 第 3 層）。「Core ATen 由来」は第 1 層の
母集団の説明（有限収束の根拠）であって判定軸ではない。

| 層                                          | 置き場                                                                   | コミットメント                                                            |
| ------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| **第 0 層 — export 時消滅**                 | `FOLDABLE_OPS` 定数畳み込み / normalize・convert の分解パス              | なし（ランタイムに届かない）                                              |
| **第 1 層 — 必須プリミティブ（原子）**      | IR 語彙（原子 ∩ Core ATen — 上限 160 → 実装対象 120）                    | 恒久・ただし有限収束（天井は増えない）。**台帳のみ（ADR 不要）**          |
| **第 1' 層 — IR 機構 op（原子・ATen 外）**  | IR 語彙（karume の次元言語・実行モデルが要求 — 初例 `sym_prefix_slice`） | **恒久の公開コミットメント**・ADR 必須                                    |
| **第 2 層 — 推奨カーネル（分子・IR 可視）** | IR 語彙 + 専用カーネル（分子の保存と attr 変種の別 op 化）               | **恒久の公開コミットメント**（全資産が全将来ランタイムに要求）・ADR 必須  |
| **第 3 層 — 推奨融合（実行時のみ）**        | `fusion.ts` の隣接 matcher（silu / rope / upsample2x / adaln）           | ゼロ（撤回自由）。exact-match が外れると黙って遅くなる — **観測点が必須** |

判定（上から順に）:

1. **値が export 時定数か**（依存の葉が arange / lifted 定数のみ）→ 第 0 層 FOLDABLE。
   条件: `_check_prefix_commutes` の 2 点評価を通ること・initializer にできるのは f32/i32 のみ。
2. **既存プリミティブの合成で厳密同値か** → 第 0 層の分解パス。**torch 突合ゲート必須**
   （上の反例集のとおり、丸め・NaN・境界で壊れる分解が多い）。
3. 合成が数値的に不可能（log1p 型）or 容量・性能で非成立（batch_norm 537MB 型）= **原子**。
   Core ATen 内 → 第 1 層。契約 1 セット（`OP_CONTRACTS` + `karume/ops.py` +
   `shapes.py` + fixtures/op-contracts.json + CPU 参照 + golden COVERAGE）。
   Core ATen 外で IR / 実行モデル自体の要求 → **第 1' 層**（ADR とセット）。
4. **分子を保存したい（合成で書けるが融合維持が要る）・attr 変種** → 第 2 層
   （**小 ADR とセット**）。入場条件は
   いずれか: ①分解形からの実行時復元が脆い / 意味情報が落ちる（attention の有限マスク値・
   rms_norm の weight 長）②中間実体化が容量・帯域で受け入れ不能 ③再出現率が恒久コミットを
   正当化する ④attrs 空契約（ADR 0012）維持のため attr 変種を別 op に分ける。
5. **正しく動く分解形が既にあり、性能だけ回収したい** → 第 3 層の融合ルール。観測点
   （`lastRunFusions` / assets_fusion_counts_test）を持つことが入場条件（ADR 0040）。

「やらない方がよいこと」（下節）は全層に優先する。

NOTE: 2026-08-11 `gelu_tanh` を第 2 層に追加（ADR 0043 の初適用 — Gemma 系の
`approximate="tanh"`。EmbeddingGemma / Gemma 4 E2B の config で使用を確認）。

NOTE: 2026-08-11 `sin`（f32 unary）を**第 1 層**に追加。根拠 = DACVAE の Snake 活性
`x + (α+1e-9)⁻¹·sin²(αx)`（Irodori-TTS v4 の codec）。手順 1 で止まらない理由は
**x が実行時値**だから — 定数の RoPE 表は従来どおり `FOLDABLE_OPS` の `aten.sin.default`
が畳み、両経路が同時に成立する（畳み込みを外すと実行時 dispatch が増える）。手順 2 の
合成も不可（`sin` を既存プリミティブで厳密に表す式が無い）。`aten.sin.default` は
Core ATen 内なので第 1 層。**`cos` は足さない** — 実行時値を取る `cos` は実測に無く、
「対称性のための追加をしない」が優先する。ADR は書かない（第 1 層は台帳のみ）。

NOTE: 2026-08-12 判定軸を**原子/分子**へ一本化（ADR 0043 追記）。境界の確定: `gelu` は
原子（erf 不在 → 第 1 層 — 分解禁止リストの所属は不変で、分解禁止は層と独立の軸）/
`leaky_relu` は分子（第 2 層・根拠 ADR 0015）/ `sym_prefix_slice` は**第 1' 層**の初例。
導入済み op の再割当はしない（層は今後の追加手続きだけを支配する）。

## 実装順序（プロトタイプ裁定のまま引き継ぎ — Karume での再裁定は ADR にて）

1. 行 reduce 族（amax/amin/max/min/argmax/argmin）— **部分消化（2026-08-12 註）**:
   実装済みは sum / amax / amin の 3 本（safe-softmax の前提分のみ。ADR 0007 決定 3 の
   「6 本を最初期に」は未充足のまま）。max / min / argmax / argmin は未実装 — argmax は
   LLM 波（greedy デコードの出口）で再訪（ADR 0043 追記 決定 4）。
2. elementwise 一括（WGSL 組込みパススルー）
3. tensor-tensor 比較の一般化
4. 軸の一般化（slice の step、pad 任意軸、~~sum/amax の dim~~ → **実装済み（2026-08-04）**:
   reduce 族（sum/amax/amin）は attrs `dim` を**宣言必須**で持ち、最終次元は行カーネル・
   それ以外は軸変種で実行する。縮約順序は両変種で厳密一致 =
   出力ビット同一（research/2026-08-04-vae-axis-reduce-recon.md））
5. エクスポータ分解パス集中投入（≈25 op、新カーネル 0）
6. native_group_norm 専用カーネル
7. conv 契約拡張（B=1 等の緩和）
8. pooling / upsample
9. scatter_add（ADR 前提）
10. 保留（sort/topk/fft/データ依存形状）

## 未決・未検証（着手前に潰す）

- ~~WGSL の NaN 比較が IEEE754 どおり false になるか~~ → **実測済み（2026-08-02）**: 比較単体は
  仕様どおり false。ただし `select(x, m, x < m)` は max イディオムへ畳まれてドライバの `max` が
  NaN を飲むため、NaN 判定はビット列で行う（根治済み — 上の反例集）
- WGSL `sign(NaN)` と torch の乖離（NaN を返す） — 分解でも組込みでも残る
- select_scatter 分解の長さ 0 スライスが WebGPU の size>0 binding 要求で落ちるか
- ~~（プロトタイプ残課題）silu が UnaryOp にあるが executor から到達不能の死枝疑い~~ →
  **解消（2026-08-08）**: 公開 op は足さず、`sigmoid → mul` の隣接 2 ノードを実行時にだけ畳む
  executor 内部の融合ルールにした（`src/runtime/fusion.ts`）

## 隣接で気づいた点（プロトタイプ側の観察）

- プロトタイプのエクスポータには定数畳み込み専用の処理経路（FOLDABLE 経路）があり、
  `arange` / `full` / `sin` / `cos` / `sign` / `ceil` / `reciprocal` 等は既にここで処理
  済みだった。「新カーネルを書く」「既存 op へ分解する」に次ぐ**第 3 の逃げ道**として、
  新しい op を足す前に必ずこの経路（Karume なら定数畳み込みパスの有無）を確認する。
- cumsum と sum は 1 スレッド=1 行の逐次走査で dim 方向の並列度がゼロ、という設計を
  プロトタイプは踏襲していた。行 reduce 族を増やす**前に**、テンプレートを逐次のまま
  複製してよいか一度裁定しておくと、後で並列化に切り替える際の書き換えが減る
  （プロトタイプはこの裁定を先送りしたまま実装を進めていた）。
