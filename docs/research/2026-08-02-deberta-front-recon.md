# DeBERTa front recon（2026-08-02）

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。

時点スナップショット — 以後の変更はこの記録を更新しない。

実測環境: torch 2.13.0+cpu / transformers 5.14.1 / Python 3.14。重みは未取得（config
のみから乱数初期化）。層数 1/2/3 で未対応 op 集合は完全一致（pre-decomp 50 op も一致）。

M1-P2（SBV2 text front の DeBERTa を Karume で動かす）の op ギャップ実測と移植計画材料。
ADR 0007 の保存 9 op を curated decompositions に入れた条件で、未対応 aten op は
**27 種**（attention_mask あり。省略時は `aten.full.default` が加わり 28 種）。
分類の内訳は 保存+融合 6 / 新カーネル 4（bmm, gather, permute 系コピー, bitwise_not）/
view メタ 3 / 分解パス 3 / 定数畳み込みで消える 11 / dtype だけ 2。

最大の構造ギャップは 3 つ。① **相対位置バケット表**（arange/sign/log/ceil/clamp/where 等）
が系列長 T 依存で、Karume の畳み込みは記号依存を構造的に拒否するため全部グラフに残る。
② **レイアウト**（view 28 / permute 22 / clone 24 / expand 5、いずれも 2 層実測）— M0 は
連続バッファのみで view 機構が皆無。③ **dtype** — 入力 i64、mask が bool、embedding/gather
の添字が整数だが M0 は意味論 f32 のみ実行可、公開 Tensor 型も Float32Array 固定。

スカラ被演算子形（`add(x, 256)` 等 10 本）と i64 elementwise（5 本）は「ハンドラはあるが
変換で落ちる」隠れギャップとして別途実測した（未対応 op の列挙には出ない）。

プロトタイプが DeBERTa に当てていた export 可能化パッチは存在しない（該当パッチは不要と
判明し削除済み。SBV2 側の別パッチは VITS front 専用で DeBERTa には触れない）。よってパッチ
有無の差分測定は該当なし。

DeBERTa は台帳の実装順序 6〜10（group_norm / conv 拡張 / pooling / scatter / sort）を
一切要求しない。

## 1. モデルの素性と再現方法

**モデル**: HF `ku-nlp/deberta-v2-large-japanese-char-wwm`（プロトタイプの export_deberta
実装が使う DEFAULT_MODEL、および SBV2 text front の export 実装が使う BERT_REPO と同一 —
SBV2 text front が使う BERT がこれである）。SBV2 のグラフ分割は
G1(DeBERTa, T) → ホスト word2ph → G2(enc_p+dp+sdp) で、DeBERTa は G1 単体
（プロトタイプの VITS front recon 記録より）。

**config**（HF から config.json のみ取得。重みは未取得）:

- model_type = deberta-v2 / num_hidden_layers = 24 / hidden_size = 1024
- num_attention_heads = 16 / attention_head_size = 64 / intermediate_size = 4096
- vocab_size = 22012 / max_position_embeddings = 512
- relative_attention = true / pos_att_type = [p2c, c2p] / position_buckets = 256
- share_att_key = true / position_biased_input = false / type_vocab_size = 0
- conv_kernel_size = 3 / conv_act = gelu / layer_norm_eps = 1e-07 / pad_token_id = 0

プロトタイプ research の記載と完全一致。

**再現方法**: `DebertaV2Config(**config.json)` → `DebertaV2Model(config)` を
`torch.manual_seed(0)` で乱数初期化（`from_pretrained` は使わない）、
`_attn_implementation="eager"`、`num_hidden_layers` を 1/2/3 に切り詰め、`eval()`。
ラッパは `forward(input_ids, attention_mask) -> out.hidden_states`（タプル返し）と、
mask 無し版 `forward(input_ids)` の 2 本。入力は `input_ids` i64[1,T]・
`attention_mask` i64[1,T]、`dynamic_shapes` は `Dim("T", min=2, max=512)` を両方の
dim1 に設定。`torch.export.export(strict=False)` → `ep.run_decompositions(curated_decompositions(preserved))`
→ `normalize_graph` → `convert`。

**環境**: torch 2.13.0+cpu / transformers **5.14.1** / Python 3.14
（`uv run --with transformers` で repo の tools/exporter 環境に一時追加。pyproject.toml /
uv.lock は未変更）。プロトタイプ実測（同 5.14.1）と pre-decomp 50 op が一致。transformers
のマイナー更新でモデリングコードが変わればグラフも変わりうるので、P2 実装時はこの
バージョンをピンして開始すべき。

**規模の目安**（2 層・ADR 0007 保存後・normalize 後）: 341 ノード / 39 op 種。24 層なら
約 3400 ノード（プロトタイプ実測 3370 と整合）。

## 2. 未対応 op 全件の分類表

ADR 0007 の保存 9 op を curated decompositions に入れ、attention_mask ありで実測した
条件下の未対応 aten op、全 27 件。

| aten                       | 分類            | 根拠note                                                                                                                                                                                                                                                                                                               |
| -------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aten.linear.default`      | preserved_fused | 2層で16本（q/k/v/attn_out/inter/out + pos_query/pos_key）。保存しないと addmm(16)+view に散る（M0 既定で実測）。融合カーネル1本が正。attrs は不要だが bias 有無の分岐が要る。                                                                                                                                          |
| `aten.layer_norm.default`  | preserved_fused | 2層で7本（embeddings 1 + rel_embeddings 1 + 層あたり2 + conv 1）。保存しないと native_layer_norm（3出力）＋operator.getitem(7) になり、IR v1 の単一出力前提と衝突する。保存は事実上必須。attrs に normalized_shape/eps が要る。                                                                                        |
| `aten.softmax.int`         | preserved_fused | 2層で2本。保存しないと aten._softmax.default（これも語彙外）になるだけで得が無い。attrs に dim（実測は -1 のみ）。safe-softmax の max 減算は行 reduce amax（M0 実装済）で組める。                                                                                                                                      |
| `aten.conv1d.default`      | preserved_fused | 1本のみ（encoder.conv、kernel 3 / stride [1] / padding [1]）。保存しないと aten.convolution.default の汎用形になる。B=1・groups=1・dilation1 の狭い契約で足りる（ADR 0007 のとおり契約はランタイム capability 側に置く）。                                                                                             |
| `aten.embedding.default`   | preserved_fused | 1本。core decomposition では分解されない（M0 既定でも embedding のまま残ることを実測）ので保存リスト云々と無関係に専用カーネルが要る。weight f32[22012,1024] × index i64[1,T] → f32[1,T,1024]。添字の整数読みが dtype ギャップ本体。                                                                                   |
| `aten.masked_fill.Scalar`  | preserved_fused | 2層で3本（attention の -3.4028234663852886e+38 埋め×2 + conv 経路の 0 埋め×1）。保存しないと where.self + scalar_tensor（scalar_tensor は畳み込みで消える）になるので、where を実装するなら分解でも可。scalar 値を載せる attrs が必須（M0 は全 op attrs 空）。                                                         |
| `aten.bmm.default`         | new_kernel      | 2層で8本（c2c 1 + c2p 1 + p2c 1 + attn·v 1 の層あたり4）。M0 の matmul は rank2×rank2 のみ。[16,T,64]×[16,64,T] 等のバッチ形。台帳の「軸の一般化」に相当。バッチ stride-0 を持たせれば後述 repeat の吸収にも効く。                                                                                                     |
| `aten.gather.default`      | new_kernel      | 2層で4本（層あたり c2p/p2c 各1）。dim=-1、src f32[16,T,512]、index i32[16,T,T]。index は畳み込み済み定数の expand なので、expand を stride-0 で読める設計なら [16,T,T] の実体化を避けられる（メモリは T=512 で 16MB 級）。scatter ではないので決定性の問題は無い。                                                     |
| `aten.view.default`        | skip            | 2層で28本。実測グラフでは view は必ず contiguous な値（clone 直後 or linear 出力）に掛かっており、要素順を変えないメタ操作のみ。値の宣言 shape を差し替えるだけで実体コピー不要。ただし「常に contiguous に掛かる」は今回の実測に基づく観測で、レイアウト機構の設計裁定と一体で確定させること。                        |
| `aten.unsqueeze.default`   | skip            | 2層で13本。長さ1の軸挿入のみでメタ操作。view と同じ扱い。                                                                                                                                                                                                                                                              |
| `aten.squeeze.dims`        | skip            | 2層で5本。長さ1の軸削除のみでメタ操作。view と同じ扱い。                                                                                                                                                                                                                                                               |
| `aten.permute.default`     | new_kernel      | 2層で22本。[0,2,1,3] / [0,2,1] のみ。実体化コピー1本（strided copy）で全部賄える。プロトタイプは (offset, strides[4]) モデルの strided codegen で permute/broadcast/slice/repeat を1カーネル族に統合していた（STRIDED_RANK=4、DeBERTa は全て rank≤4）。この前例を採るか、permute 専用の materialize にするかが設計軸。 |
| `aten.expand.default`      | new_kernel      | 2層で5本（gather 添字 [1,T,T]→[16,T,T] が層あたり2、conv の bool マスク [1,T,1]→[1,T,1024] が1）。stride 0 で読めれば実体化不要。permute と同じ strided copy 族で吸収できる。                                                                                                                                          |
| `aten.clone.default`       | decomp_pass     | 2層で24本（畳み込み済み8本を除く）。内訳は ① permute 直後の contiguous 材料化 ② eval 時 dropout 由来の恒等コピー ③ functionalization の挿入。②③ は normalize で恒等除去、① は permute のコピーに吸収される。単独の clone カーネルは不要。                                                                              |
| `aten.repeat.default`      | decomp_pass     | 2層で4本、**実測した repeat 引数は全て [1,1,1]**（share_att_key かつ B=1 のため）。恒等除去パス1本で消える。プロトタイプ research のフォローアップ「repeat の broadcast 吸収可否（P3 判断）」はこれで解消。ただし B=1 前提の観測なので、B>1 を通す日が来たら再測すること。                                             |
| `aten._to_copy.default`    | dtype_only      | 2層で7本。内訳（1層グラフで確認）: mask i64→f32 が2本、mask i64→bool が層あたり1本、相対位置表の i64→f32 / f32→i64 が2本。後者2本は畳み込みで消えるが、mask 経路の3本は入力値依存で残る。cast op（i32/i64/bool/f32 間）が必要。                                                                                        |
| `aten.arange.start_step`   | foldable        | 2本（arange(0,T) ×2）。FOLDABLE_OPS には既に載っているが、値が記号 T に依存するため convert._classify_foldable が畳むことを拒否する（convert.py の設計どおり fail loudly 側）。Tmax 畳み込み + 記号 prefix スライスを導入すれば消える。以下 foldable 判定はすべて同じ条件付き。                                        |
| `aten.sign.default`        | foldable        | 1本。相対位置バケット表（make_log_bucket_position）の中。T のみに依存し値は (i-j) の関数なので prefix スライスと可換。台帳の未決「WGSL sign(NaN) の乖離」は、畳み込みで消えるなら P2 では踏まない。                                                                                                                    |
| `aten.ceil.default`        | foldable        | 1本。同バケット表。台帳の反例集は floor/ceil/round/trunc は WGSL 組込み1行で足りると裁定済みなので、畳まない選択でも実装は軽い。ただしバケット境界の 1ulp 差が gather 添字1ずれになるバグクラスは残る（プロトタイプが焼き込みを選んだ理由がここ）。                                                                    |
| `aten.clamp.default`       | foldable        | 4本（層あたり2、c2p/p2c の添字 clamp(x+256, 0, 511)）。バケット表の下流。畳まない場合は min/max のスカラ attrs が要り、台帳の「WGSL max の NaN 非伝播」既知乖離を踏む。                                                                                                                                                |
| `aten.lt.Scalar`           | foldable        | 1本（\|rel_pos\| < 128 のバケット判定）。バケット表内。畳まない場合は比較 op + bool 値の実行系が要る（台帳の実装順序3「tensor-tensor 比較の一般化」）。                                                                                                                                                                |
| `aten.gt.Scalar`           | foldable        | 1本。lt と対（-128 < rel_pos）。同上。                                                                                                                                                                                                                                                                                 |
| `aten.le.Scalar`           | foldable        | 1本（abs(rel_pos) <= 128 の分岐）。同上。                                                                                                                                                                                                                                                                              |
| `aten.bitwise_and.Tensor`  | foldable        | 1本（bool[T,T] × bool[T,T]、バケット表内）。畳まない場合は bool の論理積が要る。                                                                                                                                                                                                                                       |
| `aten.bitwise_not.default` | new_kernel      | 2層で2本（層あたり1）。**これだけは畳めない** — attention_mask の値に依存する経路（mul → i64→bool cast → bitwise_not → masked_fill）。bool（u32 0/1 格納）の否定が実行系に必須。                                                                                                                                       |
| `aten.where.self`          | foldable        | 2本。いずれもバケット表内（bitwise_and の分岐と le の分岐）。畳めば消えるが、masked_fill を分解で処理する道を取るなら where は結局要る。3項 elementwise + bool 条件。                                                                                                                                                  |
| `aten.slice.Tensor`        | foldable        | 1本（bucket 表の [0:T] 切り出し）。これがまさに「記号 prefix スライス」そのもので、プロトタイプは専用 IR op でこれを表現していた。IR v1 に対応表現が無いのが現状の壁。                                                                                                                                                 |

（`aten.full.default` は attention_mask を渡さない場合のみ 1 本追加で出現する op で、
SBV2 front は常に mask を渡すため上表からは除外した。M0 既定条件でのみ出現する
`aten.addmm.default` / `aten._softmax.default` / `aten.convolution.default` /
`operator.getitem`（linear/softmax/conv1d/layer_norm を保存すれば出ない）、未対応列挙
自体には現れない隠れギャップ（スカラ被演算子形・i64 elementwise 形）、および既に
SKIP_OPS で処理済みの `aten.sym_size.int` / `aten.alias.default` も、次節以降で
別途扱う。）

## 3. dtype ギャップ

1. **入力の意味論 dtype**: `input_ids` / `attention_mask` はどちらも torch.int64。
   `convert.DTYPE_NAMES`（convert.py:46）は f32/i32/bool のみで **int64 が無い**ため、
   そもそも placeholder の宣言段階で fail loudly になる。裁定が要る:
   ① エクスポータ境界で i64→i32 に落とす（T≤512・vocab 22012 なので値域は安全、範囲検査
   つき）② IR v1 の意味論 dtype に i64 を足す。プロトタイプは ②（DTYPE_NAMES に i64 を
   持ち、GPU 転送時に i64→i32 変換）を採っていた。i32 で足りることは実測値域から言える
   ので ① の方が語彙を増やさない。
2. **入力転送の実行系**: `src/runtime/plan.ts:72` が入力 dtype を f32 以外拒否、
   `src/ops.ts:103` の `RUNTIME_SUPPORT.io` も M0_DTYPES（f32）から導出。ここを
   i32/bool へ広げる必要がある。加えて **公開 API の `Tensor.data` が
   `Float32Array<ArrayBuffer>` 固定**（`src/runtime/executor.ts:53-57`）で、
   `#uploadInput`（executor.ts:290 付近）も `writeBuffer(buffer, 0, tensor.data)` と
   型固定。i32 入力を渡す口が無いので **ADR 0008 の公開面の改訂**（Tensor を dtype
   判別ユニオンにする等）が伴う。バッファサイズ計算も `count * 4` なので i32/bool(u32)
   なら要素4バイトのまま流用できる。
3. **op 契約の dtype 解禁**: `src/ops.ts:63` の `M0_DTYPES=['f32']` が全 op 契約に配られ、
   `assertDtype`（ops.ts:152）が i32/bool 値を拒否する。Python 側
   `tools/exporter/karume/ops.py:27` の `M0_DTYPES` も同じ。TS/Python の契約表を
   1セットで広げる（CLAUDE.md の「op を足すときは TS 契約表・本表・golden を1セット」）。
4. **elementwise codegen の型パラメタ化**: `src/codegen/elementwise.ts` の生成 WGSL は
   `array<f32>` 固定で、正準化も UNARY_OPS/BINARY_OPS（f32 前提）のみ。i32 の mul/sub
   （マスク外積）、bool の bitwise_not、bool 条件の where/masked_fill を通すには、要素型を
   正準化キーに入れた型パラメタ化が要る（決定性の不変条件どおり、キーと WGSL を同じ
   正準化から作ること）。プロトタイプの strided codegen は
   `ScalarType = 'f32'|'i32'|'u32'|'bool'` と `cast` 式ノードでこれを実現していた
   （bool の GPU 格納は u32 0/1 が同前例の規約）。
5. **cast op が無い**: `aten._to_copy` が2層で7本。i64→f32（mask の重み掛け）、
   i64→bool（mask の真偽化）、f32→i64（バケット添字の丸め）。IR に cast op を1本足し、
   codegen で `f32(x)` / `i32(x)` / `u32(x != 0)` を出す。f32→整数の丸め規約（torch は
   truncate）を契約に明記しないと静かにずれる。
6. **embedding の添字読み**: weight f32[22012,1024] を index i32[1,T] で引く。行 gather
   なので専用カーネルは軽い（1行=1スレッド or 1workgroup）が、**添字バッファを i32 として
   読む**経路が M0 に存在しない（全バッファ f32 前提）。bind group の要素型が f32 固定で
   ある点が具体的な壁。
7. **gather の添字読み**: dim=-1 の gather、src f32[16,T,512] × index i32[16,T,T]。同じく
   整数バッファ読みが要る。添字は畳み込み定数（後述）由来なので、**i32 の initializer**
   が必要になる — 現状 `convert._add_const`（convert.py:302）と
   `_materialize_initializer`（convert.py:453）は意味論 f32 のみ、
   `docs/ir-v1.md:55` も「initializer の意味論 dtype は f32 のみ（整数重みの語彙は将来の
   改訂）」と明記。相対位置表を焼き込む方針を採るなら **IR v1 の改訂が必須**。実測でも
   lifted な i64 スカラ定数（バケット表の 0）が畳み込み経路でこの制約に当たる。
8. **mask 適用の bool 経路**: `mul`（i64 × i64 → i64[1,1,T,T] の mask 外積）→
   `_to_copy` で bool → `bitwise_not` → `masked_fill(-3.4028234663852886e+38)`。および
   conv 経路の `1 - attention_mask` → bool → expand[1,T,1024] → `masked_fill(0)`。
   **この経路だけは入力値依存で畳み込み不能**なので、bool 値の実行（u32 格納・否定・
   条件選択）は P2 で必ず要る。回避案として「mask を f32 で受けて
   `x*m + (1-m)*NEG` に書き換える」はあるが、-3.4e38 を掛ける経路が入るので golden
   突合で数値差を確認してから採ること（ADR 0007 の「分解判断は torch 突合ゲート必須」）。
9. **出力側**: DeBERTa の user 出力は hidden_states（全て f32）なので、readback が
   `Float32Array` 固定（executor.ts:515）でも P2 は通る。ただし Tensor 型を多型化する
   なら入出力で対称に決めるべきで、裁定だけは P2 で record しておくのが安全。

## 4. エクスポータ正規化・畳み込みの候補パス

1. **恒等 repeat の除去**: `aten.repeat.default(x, [1,1,1])` → x（実測4本すべてこの形）。
   プロトタイプ前例: normalize に repeat 系の書き換えは無く、IR op `repeat` として持ち込んで
   executor 側の恒等化テーブル（expand/repeat 用の IDENTITY_ALIAS）で恒等化していた。
   Karume はエクスポータ側で消すほうが IR 語彙を増やさず素直。
2. **恒等 add の除去**: `aten.add.Tensor(x, 0)` → x（実測2本、c2p/p2c 加算の初期値）。
   同時に既存の `_pow2_to_mul` と同じ粒度で書ける小パス。
3. **恒等 clone の除去**: eval 時 dropout 由来 / functionalization 挿入の
   `aten.clone.default` を、直前が contiguous な値なら削除。permute 直後の clone だけは
   レイアウト材料化なので残す（レイアウト機構の裁定に依存）。プロトタイプ前例:
   実行時の別名化テーブル（ALWAYS_ALIAS）に clone が入っており、実行時に別名化していた。
4. **スカラ被演算子の定数昇格**: `add/sub/mul/div(tensor, python_scalar)` → 二項 op +
   rank1 の f32 定数 initializer（broadcast で吸収）。実測10本。IR 語彙も attrs も増やさず
   に済む唯一の道。逆順（`sub(1, tensor)` = rsub）は「定数 − tensor」に展開する。
   プロトタイプは `aten.add.Scalar` / `aten.mul.Scalar` / `aten.rsub.Scalar` を IR op 側で
   受けていた（rsub 用の専用ハンドラと executor ケース）ので、ここは Karume 独自判断に
   なる。
5. **記号依存の定数畳み込み拡張（最重要）**: 現行 `_classify_foldable`（convert.py:245）は
   記号に触れる部分木を一律に非適格にする。プロトタイプは「Tmax で実評価 → 記号 prefix
   スライスで取り出す」方式で、**畳み込みのたびに2点評価して prefix 一致を実測**していた。
   これを入れると相対位置バケット表の arange/sign/lt/gt/bitwise_and/where/abs/log/ceil/
   clamp/le/slice/neg と付随スカラ演算が丸ごと消え、未対応 op 27 種 → 実質 13 種前後まで
   落ちる。allowlist 掲載だけでは守れない不変条件（sym_size 経由で T が「値」として入る）
   を検査で担保する設計もそのまま移植価値がある。
6. **記号 prefix スライスの IR 表現**: 上と一体。プロトタイプは
   `sym_prefix_slice`（attrs に `{sym, slices:[{dim,coeff,offset}]}`）という IR op を
   持っていた。Karume の IR v1 は次元式 `coeff·sym+offset` を既に持つので、attrs 語彙を
   解禁すれば同型で載る。**ただし i32 initializer の解禁が前提**（バケット添字は整数）。
7. **layer_norm 多出力の回避**: `native_layer_norm` + `getitem` を持ち込まないため、
   curated_decompositions の保存リストを M0 の gelu のみから **ADR 0007 の 9 op** へ拡張
   する（`convert.M0_PRESERVED_OP_PREFIXES`、convert.py:94 に既に将来拡張のコメント
   あり）。normalize パスではないが、投入順序としてはここに置くのが自然。
8. **（採らない方がよい案の記録）** masked_fill を where + scalar_tensor へ分解する道は、
   scalar_tensor が畳み込みで消えるので成立はする。ただし ADR 0007 は masked_fill を保存
   9 op に含めており、-3.4e38 埋め → softmax の経路は台帳の反例集「tensor 比較 = sub +
   scalar 比較は inf−inf=NaN で壊れる」が現実に出る箇所でもある。融合カーネルを保つ既定を
   維持し、分解案を採るなら torch 突合ゲートを通すこと。

## 5. 融合 op（保存リスト拡張）の実測内訳

- **linear**（16本/2層）— weight[out,in] × bias。DeBERTa は bias 常時あり。保存しないと
  addmm + view に散る（実測）。
- **layer_norm**（7本/2層）— normalized_shape=[1024] / eps=1e-07 / affine あり。保存しないと
  3出力 native_layer_norm + getitem になり IR v1 の単一出力前提と衝突するため、**保存が
  事実上必須**。
- **softmax**（2本/2層）— dim=-1 のみ。行 reduce amax（M0 実装済）+ exp + sum で組めるが、
  保存して1カーネルにするのが ADR 0007 の既定。
- **gelu**（3本/2層）— approximate='none'（厳密 erf 形）。**M0 で既に保存・実装済み**。
  DeBERTa は tanh 近似を使わないので追加作業なし。
- **embedding**（1本）— core decomposition が分解しないので保存リストと無関係に専用カーネル
  が要る。f32[22012,1024] × i32[1,T]。padding_idx は attrs に載るが実測は 0 で、forward
  では効かない（backward 専用）ため無視してよいか契約で明示すること。
- **masked_fill**（3本/2層）— bool マスク + f32 スカラ。attrs に値を載せる必要があり、
  **M0 の「全 op attrs 空」契約を最初に破る op** になる見込み。
- **conv1d**（1本）— in=out=1024 / kernel 3 / stride [1] / padding [1] / groups 1 /
  dilation 1 / bias あり / B=1。保存しないと汎用 convolution 形。カーネル契約の狭さは
  IR ではなくランタイム capability として表現する（ADR 0007）。
- **（保存9 opのうちDeBERTaに出ないもの）** conv2d / conv_transpose1d は出現しない。P2
  では curated の保存リストに入れておくだけでよく、カーネル実装は不要。

## 6. リスクと設計裁定が要る点

1. **【最大の設計裁定】相対位置バケット表の扱い**。表は T 依存（値も shape も）で、
   Karume の `_classify_foldable` は記号依存の畳み込みを構造的に拒否する（convert.py の
   docstring が「記号依存の焼き込みは表現できないので fail loudly 側に倒す」と明記）。
   3案: ① プロトタイプ方式＝Tmax=512 で実評価 → 記号 prefix スライス（IR v1 に
   sym_prefix_slice + i32 initializer の改訂が要る。未対応 op が 27→13 種に激減。バケット
   境界の 1ulp 差 → gather 添字1ずれのバグクラスを構造的に排除できるのが本質的な利点）
   ② GPU で整数演算をそのまま実行（IR 改訂は不要だが i32 elementwise・比較・where・
   sign・ceil・log・clamp の実装と、台帳の未決3件＝WGSL の NaN 比較 / sign(NaN) / max の
   NaN 非伝播を全部踏む）③ ホスト側で事前計算して追加入力として渡す（実装は最軽量だが、
   モデルファイル1個で完結する設計と ADR 0008 の薄い公開面に反する）。**推奨は ①**だが
   IR v1 の非互換改訂を伴うので ADR が要る。
2. **【設計裁定】レイアウト機構**。view 28 / permute 22 / clone 24 / expand 5 / squeeze 5 /
   unsqueeze 13（2層）。M0 は「連続バッファ + numel*4 確保」のみで view の概念が無い。
   案 A: プロトタイプの strided モデル（出力は連続、各入力を (offset, strides[4]) で読む
   単一カーネル族。STRIDED_RANK=4 で DeBERTa は全 rank≤4 なので収まる）。案 B:
   view/squeeze/unsqueeze は値のメタ差し替え、permute/expand/clone は実体化コピー。A は
   elementwise codegen の全面書き換えになるが、後続の SBV2 front（rank≥5 の rank 下げ
   パスが要る）まで見ると A の方が debt が小さい。**P2 の中で最も blast radius が大きい
   判断**なので先に裁定すること。
3. **【設計裁定】attrs 語彙の解禁**。M0 は TS/Python 両契約表で全 op が `attrKeys` 空。
   layer_norm(eps, normalized_shape) / softmax(dim) / conv1d(stride,padding) /
   masked_fill(value) / gather(dim) / permute(dims) / slice(dim,start,end) で attrs が
   要る。IR v1 のスキーマ自体は attrs を持つので改訂は軽いが、**契約テーブルの検証規律
   （未知 attr は fail loudly）を op ごとに設計する**必要がある。
4. **repeat=[1,1,1] は B=1 の観測**。share_att_key かつバッチ1のためで、B>1 を通す日には
   本物の repeat になる。SBV2 front は B=1 固定なので P2 では問題ないが、恒等除去パスは
   「引数が全て1のときだけ発火」と条件を明示し、それ以外は fail loudly にすること
   （黙って近似しない）。
5. **transformers 5.14.1 依存**。DeBERTa のモデリングコードが変わるとグラフ形（特に
   バケット表の構成順序と `_to_copy` の本数）が変わる。P2 開始時にバージョンをピンし、
   golden 生成と op 実装を同一バージョンで固定すること。なお DeBERTa 用の export 可能化
   パッチは torch 2.13 + transformers 5.14.1 では**不要**（プロトタイプが strict/非strict
   両モードで op 集合完全一致を実証済み、該当パッチは削除済み）。SBV2 側の別パッチは
   VITS front 専用で DeBERTa には触れない。
6. **masked_fill の埋め値が -3.4028234663852886e+38**（f32 最小有限値）。IR v1 の
   「非有限値は JSON リテラルでも値レベルでも拒否」には触れないが、attrs に f32 の極値を
   載せる形になるので JSON の往復で ulp が変わらないことを確認すること。加えて
   limitations.md の「amax/amin の identity は ±F32_MAX」との相互作用（全要素がマスク
   された行の softmax）を golden で踏むかどうか、padded ケースで検証する価値がある。
7. **gather 添字の実体化コスト**。index は expand で [1,T,T]→[16,T,T]。T=512・i32 で
   16MB/本 × 層あたり2本 × 24層。CSE で畳めば2本に落ちる（全層同一）が、expand を
   stride-0 で読めない設計にすると無駄に膨らむ。プロトタイプが expand を畳み込み frontier
   で止めていたのと同根の注意点。
8. **pos_query/pos_key が固定長 512 で計算される**。rel_embeddings[512,1024] に
   layer_norm → linear ×2 → [16,512,64]。T に依らない静的部分だが、パラメータ由来なので
   Karume の畳み込み葉（lifted 定数のみ適格）にならず毎 run 計算される。24層で linear
   48本ぶんの無駄。これは正しさの問題ではないが、性能マイルストーンで「パラメータ由来の
   静的部分木の事前計算」を検討する材料として記録しておくべき。
9. **24層フルの実測は未実施**。今回は1/2/3層で op 集合が完全一致することを確認し、
   pre-decomp 50 op がプロトタイプの24層実測と一致することを根拠に代表性を主張している。
   ノード数のスケール（24層で約3400）とメモリは未検証。

## 7. 推奨波分割（波0〜5）と台帳実装順序との整合

**波0 — 設計裁定（実装なし、ADR 2〜3本）**: ① 相対位置バケット表（Tmax 畳み込み+記号
prefix スライス／GPU 整数演算／ホスト事前計算）② レイアウト機構（strided 単一カーネル族
／メタ+実体化コピー）③ IR v1 改訂範囲（i32 initializer・sym_prefix_slice・attrs
語彙）。**この3つが後続すべての形を決めるので、実装着手前に必ず先に閉じること**。①を
採るか否かで波2以降の規模が倍違う。

**波1 — dtype 土台（中）**: i32/bool 意味論の実行解禁（`src/ops.ts` の M0_DTYPES /
RUNTIME_SUPPORT.io、`src/runtime/plan.ts:72` の入力検査、
`tools/exporter/karume/ops.py`）、公開 `Tensor` の多型化（ADR 0008 改訂）、
elementwise codegen の要素型パラメタ化（決定性キーに型を含める）、新 op は cast /
bitwise_not / where の3本、エクスポータの i64→i32 境界。台帳の実装順序「2. elementwise
一括」「3. 比較の一般化」に相当。

**波2 — レイアウト（中〜大、波0②の裁定次第）**: view/squeeze/unsqueeze のメタ扱い +
permute/transpose/expand/slice の実体化（strided copy 1本で賄うなら実質カーネル1本 + 値
メタの設計）。IR op としては 6〜7本。台帳の「4. 軸の一般化」に相当。

**波3 — 融合高位 op（大、P2 の本体）**: curated_decompositions を ADR 0007 の 9 op へ拡張
し、linear / layer_norm / softmax / embedding / masked_fill / conv1d の6カーネル + bmm
（バッチ matmul）+ gather の計8本。attrs 契約テーブルを TS/Python 同時に整備。各 op に
golden を1セットで付ける（op 語彙 allowlist 凍結の規律）。

**波4 — エクスポータ正規化・畳み込み（中）**: normalize に恒等 repeat / 恒等 add / 恒等
clone / スカラ被演算子の定数昇格の4パス。波0①で畳み込みを採るなら、記号依存畳み込み
（Tmax 評価 + 2点 prefix 一致の実測検査）と sym_prefix_slice をここに置く。台帳の「5.
分解パス集中投入」に相当。新カーネル 0。

**波5 — golden E2E（中）**: 実重みで export（2層 dev + 24層本番）→ golden 4ケース
（プロトタイプと同じ3文 + [PAD]×5 の padded ケースでマスク経路を踏む）→ Deno 側 E2E。
tolerance は既定の atol=1e-6 / rtol=1e-5。

**台帳の実装順序との整合**: 1（行 reduce 族）は M0 で実装済み、2・3 が波1、4 が波2、5 が
波4 に対応。**6〜10（native_group_norm / conv 契約拡張 / pooling・upsample / scatter_add /
sort・topk・fft・データ依存形状）は DeBERTa に一切出現しない**ので P2 のスコープ外で
確定。実装順序の裁定を変える必要はない。

**規模感**: 新規 IR op は 18〜19 本（cast, bitwise_not, where, view, squeeze, unsqueeze,
permute, transpose, expand, slice, linear, layer_norm, softmax, embedding, masked_fill,
conv1d, bmm, gather, +波0①なら sym_prefix_slice）。新規 WGSL カーネルは 8〜10 本
（elementwise/reduce の型拡張を除く）。エクスポータ側の新規パスは 4〜6 本、新カーネル 0。

## 8. 出典

- プロトタイプの DeBERTa opset 先行調査記録（24層実測 pre-decomp 50 / post-core 40、
  jit.script パッチ不要の撤回記録、上流の引数順バグ、to.dtype 内訳、repeat の未決）
- プロトタイプの export_deberta 実装（モデル id・ラッパ形状・`Dim('T',min=2,max=512)`・
  golden 4ケース［padded 含む］の前例）
- プロトタイプのコンバータ実装（FOLDABLE_OPS＝Tmax 畳み込み+prefix 可換の2点実測、
  DTYPE_NAMES に i64/f16、ATEN_HANDLERS 全表）
- プロトタイプの正規化パス実装（rank 下げ3パス・safe-softmax ガード除去・flip/split
  展開等、MAX_RANK=4 の誤爆防止線）
- プロトタイプの strided codegen 実装（ScalarType('f32'|'i32'|'u32'|'bool')・
  (offset,strides[4]) の strided elementwise モデル。bool は u32 格納の規約）
- プロトタイプの実行系実装（sym_prefix_slice / gather / embedding / masked_fill /
  i64→i32 変換）と最適化パス実装（ALWAYS_ALIAS / IDENTITY_ALIAS）
- プロトタイプの VITS front recon 記録（SBV2 front のグラフ分割 G1(DeBERTa,T) と
  BERT_REPO 特定）
- プロトタイプの SBV2 text export 実装 / SBV2 用パッチ実装（SBV2 の patch は VITS front
  専用で DeBERTa に無関与であることの確認）
- repo `tools/exporter/karume/convert.py` / `normalize.py` / `ops.py` /
  `pipeline.py` — 実測に使ったパイプライン本体
- repo `src/ops.ts` / `src/runtime/plan.ts` / `src/runtime/executor.ts` /
  `src/codegen/elementwise.ts` / `src/codegen/reduce.ts` / `src/kernels/matmul.ts` —
  dtype ギャップの具体化に読んだ実行系
- repo `docs/ir-v1.md` / `docs/op-vocabulary.md` / `docs/decisions/0007-op-vocabulary.md`
  / `docs/limitations.md` / `.claude/ACTIVE_DESIGN.md`
- 実測スクリプト: セッション作業域（揮発）。未対応 op 全件列挙+分類+dtype ヒストグラム、
  全ノードの dtype/shape 付きダンプ、ハンドラはあるが変換で落ちる形の洗い出しの3本。
  再現は本書の再現方法どおり。
- 実測ログ: セッション作業域（揮発）。1/2/3層 × mask有無 × 保存条件（M0既定/ADR0007）の
  結果 JSON、1層グラフの全ノードダンプ（178ノード）、および HF から取得した config
  （重みは未取得）。再現は本書の再現方法どおり。

---

**裁定は ADR 0009〜0012（2026-08-02）を正とする** — 本書は判断材料のスナップショット。

追記（2026-08-02・状態ポインタ）: §2 の未対応 27 種と §4 の候補パスは M1-P2 波 1〜4 で
全て消化済み（ADR 0009〜0012 と実装コミット参照。DeBERTa 1 層 export 貫通）。本文は
時点スナップショットのまま保存する。
