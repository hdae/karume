# SBV2 chain recon（front〜voice、2026-08-02）

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。

時点スナップショット — 以後の変更はこの記録を更新しない。

本 recon はプロトタイプ側の先行実験記録（research / decisions / limitations / レビュー）の
読み取りと、Karume 実装（`file:line`）との突合であり、**export の実測ではない**。実際に
torch.export を走らせた検証は M1-P3 の波 1 以降で行う。数値・行番号・件数はすべて入力記録
（プロトタイプ側の先行記録）からの転記であり、本書での新規測定・推測による数値の追加は
行っていない。

対象は SBV2（Style-Bert-VITS2 系日本語 TTS）の音響チェーン 6 モジュール —
enc_p（TextEncoder）/ dp（DurationPredictor）/ sdp（StochasticDurationPredictor, reverse）/
flow（TransformerCouplingBlock, reverse）/ dec（HiFi-GAN Generator）/ voice（flow+dec 融合）。
テキスト側（DeBERTa）は別記録（[2026-08-02-deberta-front-recon.md](2026-08-02-deberta-front-recon.md)）
の対象で、本書では前提としてのみ参照する。

## 1. モジュール構成と export 戦略

### front（enc_p / dp / sdp）

- **enc_p（TextEncoder 本体）**: phones/tones/language の embedding 3 本 + bert_proj
  （Conv1d 1x1）+ style_proj（Linear）を足し合わせ、窓付き相対位置注意 Encoder（6層・
  hidden192・heads2・kernel3・window4）を通して h と stats（→ m_p / logs_p）を出す。
  実装根拠は style_bert_vits2 パッケージ `models_jp_extra.py:362-437` /
  `attentions.py:57-124`。単独グラフとしては export されず、プロトタイプ `patch_sbv2.py:451-492`
  の融合ラッパ `Sbv2Front` が enc_p.forward を写経（x_mask 生成と `torch.split` を避けるため）
  している。
- **dp（DurationPredictor）**: h と g から logw_dp を出す。conv1d(k=3,pad=1) → relu →
  LayerNorm → conv1d(k=3,pad=1) → relu → LayerNorm → proj(1x1)、全て x_mask 乗算つき
  （`models_jp_extra.py:279-329`）。`Sbv2Front.forward` の最終行で `self.dp(h, x_mask, g=g)`
  としてそのまま呼ばれ（プロトタイプ `patch_sbv2.py:491`）、**パッチ不要**。プロトタイプの
  先行 recon は「無改造で完走（動的 T export + convert、relu 1 op 追加のみで IR 19 ノード）」
  と記録している。3 モジュール中いちばん軽く、Karume の現行語彙で**そのまま通る唯一の
  モジュール**（conv1d は stride1/padding1/groups1/dilation1、layer_norm は最終次元、
  relu/mul/add のみ）。
- **sdp（StochasticDurationPredictor, reverse）**: h と外部ノイズから logw_sdp を出す。
  pre/cond/proj(1x1 conv) + DDSConv（depthwise dilated）→ flows 逆順で
  ElementwiseAffine / Flip / ConvFlow（rational-quadratic spline）
  （`models_jp_extra.py:163-277`, `modules.py:86-133,386-435,507-573`）。プロトタイプ
  `patch_sbv2.py:418-448` の `SdpReverseNoiseIn` がラップし、原 forward の reverse 分岐と
  等価な経路を再実装。差分は `torch.randn(B,2,T)*noise_scale` を外部入力 `z_noise` に昇格
  した 1 点のみ（noise_scale 乗算はホスト側）。加えて spline を分岐フリー・非破壊の同値
  実装へ monkeypatch している。reverse で実際に走る flows は
  `[Flip, ConvFlow, Flip, ConvFlow, Flip, ConvFlow, Flip, ElementwiseAffine]`
  （`models_jp_extra.py:274` の `flows[:-2] + [flows[-1]]` をプロトタイプ `patch_sbv2.py:443`
  が踏襲）。Karume との差分がいちばん大きいモジュール。
- **Sbv2Front（融合ラッパ）**: enc_p + dp + sdp(reverse) を 1 グラフに融合し
  `(logw_sdp, logw_dp, m_p, logs_p)` を返す（プロトタイプ `patch_sbv2.py:451-492`）。動的
  `Dim("P", min=2, max=sym_max)`、sym_max 既定 512。融合の裁定根拠は「hidden の readback
  往復排除・重み 1 ファイル約 31MB」。sdp_ratio 混合と durations 化はホスト側で、**実行時
  ノブをグラフに焼かない**。プロトタイプのエクスポータ内にはモジュール別ラッパも保持されて
  いるが、実際に emit する CLI 経路は融合 front のみ。
- **export_sbv2_text.py（対象外）**: モデルグラフには一切触れず、DeBERTa 文字トークナイザ
  語彙・clean_text 判定表・style_vec(1x256)・話者埋め込み g(1x512x1) の実行時アセットと、TS
  テキスト層のパリティ検証フィクスチャのみを出す。「G1/G2/G3 のグラフは export_deberta.py /
  export_sbv2.py が出す。こちらが出すのは『テキスト → front 入力』に要るホスト側の資産だけ」
  と明記されている。

### flow / dec / voice

- **flow（TransformerCouplingBlock reverse）**: z_p → z の正規化フロー逆変換。
  `[TransformerCouplingLayer(mean_only=True) + Flip] × 4` を reversed 順に適用。coupling 内
  は front と同じ window=4 相対位置注意 Encoder（ch192/hidden192/filter768/heads2/layers6/
  kernel5/gin512）。プロトタイプのラッパ `FlowReverse(z_p, y_mask, g, idx_k, valid)` を
  `torch.export(strict=False)`、`Dim("T", min=2, max=4096)` で export。相対位置注意の
  (T,T) 表 `idx_k`(i64)/`valid`(f32) はグラフ入力に昇格し、ホスト（JS/TS）側が実 Ty で
  生成する鏡像実装を持つ。flows のループは Python 側で展開され、IR は完全に静的な直列
  グラフ。IR 換算 1803 ノード、記号次元は T と T+8 のみ。全値 rank ≤ 4。
- **dec（HiFi-GAN Generator）**: z·y_mask → 波形 (1,1,512T)。conv_pre(k7) + cond(1x1) →
  `[leaky_relu → ConvTranspose1d → ResBlock1×3 の平均] × 5`（up_rates [8,8,2,2,2] / up_k
  [16,16,8,2,2] / init_ch 512）→ leaky_relu → conv_post(k7, bias 無し) → tanh。`net_g.dec`
  を直接 `torch.export(args=(x,g), dynamic_shapes=({2:T},{}), sym_max=4096)`。前処理は
  `remove_weight_norm`（冪等）のみで、パッチ層は不要。**最終段は iSTFT ではなく
  conv_post→tanh で波形直出力**（`models_jp_extra.py:607-611`）— STFT/iSTFT 系 op は一切
  不要。post-decomp は leaky_relu 96 / conv1d 93 / add 56 / convolution(transposed) 5 /
  div 5 / tanh 1 の 6 種のみ、ノード 454。ConvTranspose の `pad=(k-u)//2` により出力長は
  厳密に L·u。
- **voice（G3 融合＝flow+dec）**: flow reverse → z·y_mask → dec を 1 グラフに融合したもの
  （参照 infer 末尾 `models_jp_extra.py:1156-1158` と同順）。`Sbv2Voice(z_p, y_mask, g,
  idx_k, valid)` を FlowReverse と同じ dynamic_shapes で export。ゴールデンは flow / dec
  単体と融合の 3 系統を出す。融合すると中間 z の readback 往復が消えるが、ランタイム側は
  「readback はグラフ出力と入力のみ」なので中間 z のデバッグ突合は単体グラフ側でしか
  できない。段階検証のため flow/dec 単体 export も残す設計。

## 2. op ギャップ表

status 列は `supported` / `missing_kernel` / `missing_contract` / `normalization_needed` /
`unknown`。supported 群はモジュールごとに 1 行へまとめている。

| モジュール            | op                                                                                                                                                                                                                                                                                                 | status                                                  | 根拠                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| enc_p, sdp            | `aten.slice.Tensor`（IR `slice`）                                                                                                                                                                                                                                                                  | missing_contract                                        | enc_p の `stats[:, :c]` / `stats[:, c:]`（プロトタイプ `patch_sbv2.py:488`）、sdp の `z[:, :1]` や spline 内 `bin_locations[..., :-1]` 等（同 44-56, `modules.py:563-566`）。Karume は `docs/decisions/0011-layout-strategy.md:51-53` で slice を見送り済み。`sym_prefix_slice` は attrs に sym・coeff≥1 を要求する記号 prefix 専用（`src/ops.ts:389-421`）で、静的・非 0 開始のチャネル slice は表現できない。                                                    |
| sdp                   | `aten.split_with_sizes` → getitem                                                                                                                                                                                                                                                                  | normalization_needed                                    | ConvFlow / ResidualCoupling の `torch.split(x, [half]*2, 1)`（`modules.py:552,556`）。プロトタイプは normalize の split→slice パスで潰すが、Karume の normalize（`tools/exporter/karume/normalize.py:215` `normalize_graph`）に該当パスは無く、落とし先の slice 自体も無い。                                                                                                                                                                                       |
| sdp, flow             | `aten.cat.default`（IR `cat`）                                                                                                                                                                                                                                                                     | missing_kernel（sdp）/ missing_contract（flow）         | spline の cat 系（プロトタイプ `patch_sbv2.py:92-94,103-105,193`）、ConvFlow 出力 `torch.cat([x0,x1],1)`（`modules.py:570`）、flow の coupling reverse `torch.cat([x0,x1],1)` 4 本（`patch_sbv2.py:542`）。Karume の契約表（`src/ops.ts:597-674`）に cat は無い。                                                                                                                                                                                                  |
| sdp                   | `aten.cumsum.default`（IR `cumsum`）                                                                                                                                                                                                                                                               | missing_kernel                                          | spline の cumwidths/cumheights（`torch.cumsum(widths, dim=-1)`、`patch_sbv2.py:90,101`）。Karume の REDUCE_OPS は sum/amax/amin のみ（`src/ops.ts:42`）。                                                                                                                                                                                                                                                                                                          |
| sdp                   | `aten.where.self`（IR `where`）                                                                                                                                                                                                                                                                    | missing_kernel                                          | spline の区間外復帰・softplus 分解形（`torch.where(inside, ...)`、`patch_sbv2.py:210-211`）。条件が入力値依存で畳めない。Karume は where を FOLDABLE_OPS（定数畳み込み専用）にのみ持ち実行系ハンドラが無い（`tools/exporter/karume/convert.py:196` vs `1250-1285`）。                                                                                                                                                                                              |
| sdp                   | `aten.clamp.default`（実行時）                                                                                                                                                                                                                                                                     | missing_kernel                                          | spline の定義域クランプ（`patch_sbv2.py:195`）。Karume の clamp も FOLDABLE_OPS のみ（`convert.py:191`）。min/max 二項も語彙に無い（BINARY_OPS は add/sub/mul/div のみ — `src/ops.ts:38`）。                                                                                                                                                                                                                                                                       |
| enc_p, flow           | `aten.eq.Scalar`（IR `eq_scalar` 相当）                                                                                                                                                                                                                                                            | missing_kernel（enc_p）/ normalization_needed（flow）   | `scores.masked_fill(attn_mask == 0, -1e4)`（`patch_sbv2.py:350`）。attn_mask はグラフ入力由来で畳めない。Karume に eq は FOLDABLE_OPS にも無い（`convert.py:186-196` は lt/gt/le の Scalar のみ）。flow 側は「`bitwise_not(cast(attn_mask,'bool'))` が同値になり得るので新 op 不要、normalize パスで済む見込み」と記されるが**実グラフでの検証は未実施**。                                                                                                         |
| sdp                   | `aten.ge.Scalar` / `aten.le.Scalar`（実行時）+ `aten.bitwise_and.Tensor`（実行時）                                                                                                                                                                                                                 | missing_kernel                                          | spline の `inside = (inputs >= -tail_bound) & (inputs <= tail_bound)`（`patch_sbv2.py:189`）。ge は FOLDABLE_OPS にも無く、bool の実行系論理積も無い（bitwise_not のみ — `src/ops.ts:34,248`）。                                                                                                                                                                                                                                                                   |
| sdp                   | `aten.ge.Tensor` + bool 上の `sum`                                                                                                                                                                                                                                                                 | missing_kernel                                          | searchsorted_free の `torch.sum(inputs[..., None] >= bl, dim=-1) - 1`（`patch_sbv2.py:57`）。Karume の sum は f32 専業（DTYPES 表に sum が無く F32 既定 — `src/ops.ts:263-276`）で bool 入力を受けない。                                                                                                                                                                                                                                                           |
| sdp                   | `aten.log1p.default` + `aten.gt.Scalar`（実行時）                                                                                                                                                                                                                                                  | missing_kernel                                          | `F.softplus(unnormalized_derivatives)` の分解形（`patch_sbv2.py:97`）。softplus は default_decompositions で `where(scaled>threshold, a, log1p(exp(scaled))/beta)` に落ちる。Karume の UNARY_OPS に log1p は無い（`src/ops.ts:24-35`）。                                                                                                                                                                                                                           |
| enc_p, flow           | `aten.constant_pad_nd.default`（IR `pad`）                                                                                                                                                                                                                                                         | missing_kernel（enc_p）/ missing_contract（flow）       | 相対位置注意の value 側 `F.pad(p_attn, [w, w])`（`patch_sbv2.py:359` / flow は同 24 層 × 1 本）。出力次元が T+2w=T+8。Karume の契約表に pad は無い。                                                                                                                                                                                                                                                                                                               |
| sdp, flow             | `aten.flip.default`                                                                                                                                                                                                                                                                                | normalization_needed（sdp）/ missing_contract（flow）   | modules.Flip（`modules.py:410`, プロトタイプ `patch_sbv2.py:570`）。チャネル数 2 の静的軸。プロトタイプは normalize の flip→cat で slice×2+cat に落とすが、Karume には flip ハンドラも落とし先の slice/cat も無い。                                                                                                                                                                                                                                                |
| sdp                   | conv1d の groups（depthwise, groups=192）                                                                                                                                                                                                                                                          | missing_contract                                        | DDSConv 3 層 + ConvFlow 3 本×3 層 = 計 12 本（`modules.py:104-112`）。Karume は attrs に欄自体を持たず（`src/ops.ts:537-546`「groups/dilation は attrs に無い＝1 固定」）、エクスポータも `_expect(groups==1, ...)` で拒否（`tools/exporter/karume/convert.py:1246`）。                                                                                                                                                                                            |
| sdp                   | conv1d の dilation（1/3/9）                                                                                                                                                                                                                                                                        | missing_contract                                        | 同 DDSConv、`dilation = kernel_size**i`（`modules.py:102-103`）。Karume は `_expect(dilation==1, ...)`（`convert.py:1245`）。                                                                                                                                                                                                                                                                                                                                      |
| enc_p, flow           | expand の f32 解禁                                                                                                                                                                                                                                                                                 | normalization_needed（enc_p）/ missing_contract（flow） | 相対位置埋め込みの 4D 化 `ek.expand(...)` / `ev.expand(...)`（`patch_sbv2.py:343,366`）。Karume の expand は I32_BOOL のみ（`src/ops.ts:271`、根拠コメント 249-254「f32 の expand は実測に現れないので解禁しない」）。カーネルは strided コピー族の共用なので dtype 解禁のみで足りる見込み（推測）。                                                                                                                                                               |
| dec                   | `aten.leaky_relu.default`                                                                                                                                                                                                                                                                          | missing_kernel                                          | ups 前 5 / ResBlock 内 90 / 最終 1 = 96 ノード。slope は 2 種混在（`models_jp_extra.py:597` の 0.1 と `:607` の既定 0.01）。Karume の UNARY_OPS（`src/ops.ts:24-35`）に無い。                                                                                                                                                                                                                                                                                      |
| dec                   | `aten.conv_transpose1d.default`                                                                                                                                                                                                                                                                    | missing_contract                                        | ups 5 本。Karume は分解抑止リストには載せている（`tools/exporter/karume/convert.py:217`）が、契約表にもカーネルにも無い — 同 212-213 が「実測に出ないのでカーネルが無い。出たら未対応 op として落ちる」と明記。                                                                                                                                                                                                                                                    |
| dec                   | conv1d の dilation（3/5）                                                                                                                                                                                                                                                                          | missing_contract                                        | ResBlock1 の convs1（`dilation=(1,3,5)`、`modules.py:249-271`）、計 30 本。Karume は attrs に欄が無く export 段で拒否（`convert.py:1245`）、出力 shape 式 `span = L + 2p - K` も dilation 非対応（`src/ops.ts:1222`）。                                                                                                                                                                                                                                            |
| dec                   | conv1d の bias 無し形                                                                                                                                                                                                                                                                              | normalization_needed                                    | conv_post（`Conv1d(ch,1,7,1,padding=3,bias=False)`）1 本（`models_jp_extra.py:583`）。Karume はアリティ 3 固定で bias 無しを export 段で拒否（`convert.py:1234-1238`、契約は `src/ops.ts:641`）。                                                                                                                                                                                                                                                                  |
| flow                  | matmul（rank-4）                                                                                                                                                                                                                                                                                   | unknown                                                 | scores/output/rel_local の 4D×4D（`patch_sbv2.py:328,343,353,366`）。Karume の matmul は rank-2 のみ・bmm は rank-3 のみ（`src/ops.ts:954-990`）。core 分解が aten.matmul を view→bmm→view に落とすため DeBERTa front では rank-3 bmm として通っており、flow でも同経路になる見込みだが**未実測**。b=1・h=2 は静的なので reshape で潰せる。                                                                                                                        |
| flow                  | `full_like`                                                                                                                                                                                                                                                                                        | normalization_needed                                    | mean_only の logs=zeros 由来。プロトタイプはパッチ層で `exp(-logs)=1` の乗算を畳んで消している（`patch_sbv2.py:531-533`、ビット一致を実測）。同じパッチを持ち込めば op 追加不要。                                                                                                                                                                                                                                                                                  |
| dec                   | div（スカラ被除数）                                                                                                                                                                                                                                                                                | supported                                               | `xs / self.num_kernels` 5 本（`models_jp_extra.py:606`）。Karume は `_promote_scalar_operands` でスカラを定数テンソルへ昇格する正規化を持つ（`tools/exporter/karume/normalize.py:144`）。                                                                                                                                                                                                                                                                          |
| enc_p/dp/sdp/flow/dec | masked_fill.Scalar / softmax / layer_norm / linear / embedding / conv1d(基本形) / bmm / relu / gelu / sigmoid / tanh / exp / log / sqrt / neg / abs / add / sub / mul / div / view / unsqueeze / squeeze / permute / pow(指数2, mul へ正規化) / gather(rank4・最終次元) / 相対位置表の定数畳み込み | supported                                               | Karume の ATEN_HANDLERS（`convert.py:1250-1285`）と契約表（`src/ops.ts:597-674`）に全て存在。gather は先行次元一致・最終次元だけ自由の契約に適合（`src/ops.ts:991-1013`）。pow(x,2) は normalize が mul(x,x) へ落とす（`tools/exporter/karume/normalize.py:44-74`）。相対位置表の arange/sub/add/abs/clamp/le.Scalar/_to_copy は全て FOLDABLE_OPS に載り、値が i-j（P 非依存）のため 2 点評価検査を通る（推測だがプロトタイプ `patch_sbv2.py:276-277` も同根拠）。 |

## 3. 構造ギャップ

- **slice 前提の崩れ**: Karume は ADR 0011 の実装時追記で slice を意識的に見送っている。
  根拠は「実測グラフの slice は相対位置バケット表の部分木 1 本のみで、ADR 0010 の Tmax
  畳み込みで消える側」だが、SBV2 chain ではこの前提が崩れる — enc_p の m_p/logs_p 分割
  だけで静的・非 0 開始のチャネル slice が要り、sdp・flow では split / spline / coupling
  reverse の各所で多用される。`sym_prefix_slice` は記号 prefix 専用で代替にならない。strided
  カーネルの params は既に `(offset, strides[4])` を持つため、slice を足す場合の可変点は
  offset 1 語のみという見通しは ADR 0011 が自ら書いている（`docs/decisions/0011-layout-strategy.md:50-56`）。
- **cat・pad の実行モデル**: Karume の現行カーネルは全て「1 ノード＝出力バッファ全域を
  書く」形（strided コピーも出力は常に連続で全域 — `src/codegen/strided.ts:4`）。cat と pad
  は複数入力/ゼロ領域を出力の部分領域へ書き込む形で、実行モデル側の拡張が要る。プロトタイプ
  は「WebGPU バッファのゼロ初期化保証＋copy_region」で新規 WGSL をゼロにし、cat も同カーネル
  の部分領域コピーで組んだが、この保証はバッファプール導入後に条件付きで偽になる（プール
  再利用バッファはゼロ初期化されない）罠つき。Karume は ADR 0004 でバッファプールを既に
  持つため、pad を足す時点でこの不変条件（出力は noReuse で新品確保する必要がある）の明文化
  が要る。
- **conv1d の契約 4 点セット**: Karume は「表現が無いので『1 以外を黙って 1 で実行する』
  経路が構造的に存在しない」ことを設計上の価値として明記している。sdp の DDSConv
  （depthwise groups=192、dilation 1/3/9）・dec の ResBlock1（dilation 3/5）・dec の
  conv_transpose1d・conv_post（bias 無し）を通すには、TS 契約表・Python 契約表・WGSL
  カーネル・CPU 参照の 4 点セットを同時に広げる必要がある（op を足すときの規律として台帳に
  既定済み）。なお padding は `2·pad = dilation·(K−1)` を満たす（`(3d−d)//2 = d`）ので出力長
  は L 固定に保たれる。
- **rank≥5 は front / flow / dec のいずれにも出ない — 障害は rank ではなく shape 式**:
  enc_p/dp/sdp を通して最大 rank は 4（注意の [b,h,P,P] と spline の [b,c,t,bins]）、
  flow/dec も全て rank≤4（flow の attention scores (1,2,T,T) が最大、dec は全 rank-3）。
  プロトタイプの rank 下げ正規化パス（unit expand / split unbind / reshape-permute）は
  DiT の GQA/RoPE/patchify 向けで、SBV2 とは無関係と明記されている。したがって ADR 0011 の
  「rank≥5 はエクスポータの rank 下げ正規化で潰す前提」は SBV2 chain では発動しない。front
  側の真の障害は enc_p の相対位置シフトが作る **2P−1 / 2P+9 / 2P² / P(2P−1) の shape 式
  144 箇所**で、二次式を含むため次元言語のアフィン拡張でも救えない — gather 化パッチが
  必須の理由がここにある。
- **相対位置表の入力昇格**: front（P≤512）は表を Tmax 焼き込み＋sym_prefix_slice で処理
  できるが、flow は Ty が数百〜数千で idx_k+valid の焼き込み定数が sym_max=4096 のとき
  134MB になる。裁定はパッチ層で Encoder/coupling の forward に表をスレッディングし、ホスト
  （JS/TS）が実 Ty で生成する方式。これは op 追加では済まない構造変更で、①Python の表構築と
  TS 側生成関数が式レベルで一致していること ②golden にも表入力の生成（Python 側鏡像）が
  要ること、の 2 つが新しい検証責務になる。表は clamp 済み（`idx_k = clamp(rel+w, 0, 2w)`）
  で value 側の idx_v も pad 済み長に対し常に範囲内なので、Karume の「gather の範囲外
  添字は NaN 汚染」規約（`src/kernels/gather.ts:11-31`）には抵触しない。
- **weight_norm 除去の順序**: weight_norm は dec 95 件（+ enc_q 33 件）に集中し、
  enc_p/sdp/dp/flow には 0 件と実測されている。SBV2 の推論経路は remove_weight_norm を
  呼ばないが export 前に呼ぶ方針で eager 出力はビット一致した。f16/i8 の重み丸めは実効重み
  に当てる必要があるため、remove を丸めより先に実行する順序制約がある（Karume 側でも
  同じ順序規律が要る）。flow に導出パラメータが無いことも丸めの前提として assert で固定
  されている。
- **P+8 次元（pad 由来のオフセット付き次元）**: value 側の `F.pad(p_attn, [w,w])` が長さ
  P+2w = P+8（flow は T+8）の次元を作る。Karume の次元言語は `coeff·sym+offset`
  （coeff≥1・offset≥0）を既に受理するので構造的には対応済み（`src/format/dims.ts:1-8,39-47`）。
  ただし sym_prefix_slice の 2 点評価検査は「オフセット付きは検査を通ったものだけ受理」と
  いう一般化が要る（プロトタイプの decisions 記録も「pad が末尾に値を置く形は落ちる」と
  副作用まで記録している）。
- **融合 1 グラフかモジュール別かの裁定**: プロトタイプは front を enc_p+dp+sdp 融合 1
  グラフ、voice を flow+dec 融合 1 グラフとして採択（hidden/z の readback 往復排除）しつつ、
  エクスポータ内にモジュール別ラッパも保持し、ゴールデンはモジュール別＋融合の両方を出す
  方針を裁定した。ただし実際に emit する CLI 経路は融合グラフのみ。Karume で段階的に通す
  なら front は dp 単体（現行語彙でそのまま通る）→ enc_p → sdp の順、後半は dec 単体
  → flow → voice の順に切れる形をエクスポータ側で用意する価値がある。
- **バッファ寿命管理は既に埋まっている**: プロトタイプの executor は素朴全実体化で run
  終了まで一切解放せず、dec の活性は素朴総和 6.97MB×Ty（Ty=200 で 1.4GB）で「素朴なままでは
  デモ不成立」と判定された。最終消費者ベースの解放＋サイズクラス別プールで縮小されている。
  Karume はアリーナ＋最終消費者カウントを実装済み（`src/runtime/executor.ts:430-437,500-510`、
  ADR 0004/0011）なので、この構造ギャップは既に埋まっている。残るのは dec 実測での peak
  検証のみ。
- **動的軸は front=P と P+8、flow=T と T+8、dec=c·T（8〜512T、252 箇所）で、係数付き次元は
  Karume 側で既に解決済み**: プロトタイプ側は c·T 対応を新規に入れる必要があり、係数付き
  入力の誤束縛バグを抱えていたが、Karume は次元言語が coeff·sym+offset を正準文法として
  持ち、シンボル束縛を「素の形（係数1・オフセット0）で現れる入力次元からのみ」取り、係数
  付き次元は 1 巡目でスキップして 2 巡目で全次元を再評価・照合する設計（`src/runtime/plan.ts:214-236`、
  `src/format/ir.ts:352-366`）のため、この負債は最初から回避している。

## 4. 罠（pitfalls）

front（enc_p/dp/sdp）と flowdec（flow/dec/voice）の両記録に出た pitfalls を重複統合して
列挙する。

- **【撤回された前提】gather の範囲外添字規約は不要**: recon 時点（2026-07-31）の gather
  書き換え骨子は「範囲外 index→0」規約に依存する初のケースとして規約明文化を求めていたが、
  最終裁定はこれを**却下**した。理由は「torch 側オラクルで同義を表現できず、P 依存マスクは
  2 点評価検査が正しく拒否する形にしかならない」。採択形は key 側＝clamp した idx_k と
  0/1 valid マスクの乗算、value 側＝[w,w] ゼロパディング（idx_v は常に範囲内）。Karume に
  gather の OOB 規約を持ち込む必要は無い。
- **4D×2D matmul の罠**: 相対位置埋め込みを 2D のまま matmul すると (b·h·P, kc) の view に
  分解され、係数付きシンボル次元 2P が生える。埋め込み側を 4D に expand して 4D×4D（静的
  batch の bmm）に落とすのが正しい回避。この 1 箇所を見落とすと次元言語の係数拡張が要るよう
  に見えてしまう。
- **【沈黙誤値クラス】相対位置注意の窓幅 w ハードコード**: front 側は w=4 が 3 箇所、flow
  側も w=4 がホスト側生成関数に焼き込まれている。モデルの window_size と食い違うと idx_k が
  `clamp(rel+4,0,8)` のまま幅 2·window_size+1 の埋め込みを gather するため**shape エラーに
  ならず黙って誤った埋め込みを読む**。しかもホスト供給表を使う経路ではゴールデンごと同じ
  誤りを共有して検証をすり抜けうる。現行はロード時の assert（window_size==4）で fail loudly
  化するのが最小対処。
- **【pad のゼロ初期化】**: pad をゼロ初期化バッファ＋copy_region で実装する場合、その前提
  はバッファプール導入後に条件付きで偽になる（再利用バッファはゼロ初期化されない）。pad の
  出力は noReuse で新品確保する必要がある。Karume は ADR 0004 でプールを先に持っているの
  で、pad を足す時点でこの不変条件を明文化すること。
- **【パッチのプロセス汚染】**: パッチはクラス属性のプロセス全域差し替えなので、「パッチ前
  の参照」を採れるのは 1 プロセスにつき 1 回だけ。--verify / --verify-flow / --verify-voice
  を併用すると後発の参照が既にパッチ済みになり、同値検証が**恒真化して偽 PASS** する
  （レビュー記録の指摘）。--verify-dec も remove_weight_norm で参照を汚染するため voice とは
  排他。dtype 丸め（f16/i8）も未丸めの参照を汚染する。プロトタイプの export スクリプトは
  ckpt ロード前に排他をチェックする形で塞いでいる。Karume 側でも同値検証ハーネスを移植
  するならこの排他が必要。
- **【verify の順序依存】**: 同値検証は「全ケースの参照値を先に確定 → パッチ適用 → 比較」の
  順序が必須。
- **【--sym-max のターゲット別使い分け】**: front=512（P=音素数）、flow/dec/voice=4096
  （T=フレーム数）。機械的強制は無く、誤値は沈黙する。焼き込み定数量がターゲットで桁違いに
  なる（front の相対位置表は Pmax=512 で数 MB だが、flow は O(Tymax²) で 4096 なら実測
  134MB）。取り違えると焼き込み定数のサイズか Tmax 超過エラーで表面化する。
- **【weight_norm は front に無い・撤回された課題】**: 先行 recon は当初 weight_norm 除去
  パッチを課題としていたが**撤回**され、legacy weight_g/weight_v は dec 95 件＋enc_q 33 件
  のみ、enc_p/sdp/dp/flow は 0 件と実測された。front/flow で weight_norm 対応を先回りしない
  こと（assert_no_weight_norm が fail loudly で固定）。remove_weight_norm のビット一致は
  1 ケース（z=(1,192,50)）の実測でありスペック保証ではない（§6 参照）。
- **【FFN pad 畳み込みの前提】**: kernel_size が奇数かつ causal=False のときだけ
  `_same_padding` は `conv1d(padding=(k−1)//2)` と厳密に等価。偶数 kernel は左右非対称
  パディングになるので assert で落とす。
- **【spline パッチの無ガード分岐】**: `piecewise_free` の `tails is None` 分岐は原実装の
  `InputOutsideDomain` 検査を削除した spline へクランプ無しで直行する形（安全弁だけが外れた
  形）。SBV2 の呼び出し元は `tails="linear"` 固定で到達不能だが、移植時に握り潰さず
  RuntimeError のまま残すこと。
- **【数値許容差の基線】**: front のパッチ前後 eager 同値は P=2..39 でビット一致、worst
  8.6e-6 @P=64（value 側縮約長変更に伴う BLAS ブロッキング順序差）。flow は T=50 ビット一致
  / T=137 max 1.19e-6。ゴールデンは**パッチ適用後**の eager 出力（＝IR が計算すべき数の
  正）で、パッチ前参照との差は --verify が別途実測する、という二層構造。
- **【spline の monkeypatch 差し替え先】**: `modules.py` が
  `from ...transforms import piecewise_rational_quadratic_transform` で関数オブジェクトを
  束縛済みのため、transforms 側を差し替えても効かない。差し替え先は
  `style_bert_vits2.models.modules`。
- **【full_like/zeros_like を IR に持ち込まない】**: 定数列は `t[..., :1] * 0.0 + value`
  で作る（t が有限値なら `x*0.0 == 0.0` が厳密に成立）。同様に mean_only な coupling の
  logs=0 は `exp(-logs)=1` の乗算ごと畳む。語彙を増やさないための定型手筋（flow の full_like
  もこれで消える）。
- **【非破壊化の妥当性根拠】**: searchsorted の in-place 破壊を除いてよいのは、破壊された
  最終要素が後段の gather（添字 ≤ num_bins−1）から参照されないため。移植時にこの理由を
  コメントごと運ばないと「同値でない変更」に見える。
- **conv_transpose1d の位置引数省略**: 保存形で位置引数の末尾既定値が省略される（ups3/4 は
  `(input, weight, bias, [stride])` の 4 引数形が実在）。ハンドラ側で padding=0 /
  output_padding=0 / groups=1 / dilation=1 を補完しないと IndexError か誤った既定で通る。
- **conv_transpose1d の重みレイアウト**: `[Cin, Cout, K]` で conv1d の `[Cout, Cin, K]` と
  は転置。取り違えても要素数が合う形が作れるため shape 検査を素通りする。
- **conv_transpose1d カーネルの stride=0 ハング**: プロトタイプ実装は stride=0 で kk が進ま
  ず**GPU ハング**（例外ではない）。呼び出し側で stride ≥ 1 を保証する MUST。
- **leaky_relu の NaN 伝播**: `max(a, slope·a)` ではなく select 形で書く必要がある。torch
  は `leaky_relu(NaN)=NaN` だが WGSL の `max` は NaN 伝播を保証しない（relu/clamp が既に
  同じ乖離を抱えていると既知化されている）。
- **leaky_relu の分解禁止**: leaky_relu を分解させると `gt_scalar+mul+where` になり中間が
  1.5〜2 倍に膨らむ。PRESERVED に入れて分解抑止することがメモリ見積の前提。
- **leaky_relu の slope 混在**: dec の leaky_relu は slope が 2 種混在する（ups/ResBlock は
  LRELU_SLOPE=0.1、最終段は torch 既定 0.01 で位置引数ごと省略）。attrs に negative_slope
  を持たせないと片方が黙って誤る。
- **tanh のビット非一致**: WGSL の tanh はブラウザ実装依存で `Math.tanh` とビット一致しない。
  dec の最終段が tanh なので、出力の突合は必ず許容誤差で行う。
- **活性メモリの実測余地**: dec の活性は素朴総和で 6.97MB×Ty（Ty=200 で 1.4GB）。flow 側も
  attention スコア (1,2,Ty,Ty) が 24 層合計で Ty=1000 のとき 192MB（層内ピークは pad 版込み
  で 2〜3 倍）。プールが効いていることを Ty 大で実測する必要がある。
- **flow の重みサイズ**: f32 で 158MB（FFN conv が 141MB）。配信は f16 格納で半減できる。
- **dec の conv_post bias 無し**: Karume の「bias 常時あり（アリティ 3 固定）」契約と正面
  衝突する。プロトタイプ流のゼロ bias 合成でアリティを正規化するか契約を緩めるかの判断が
  要る（カーネルに arity 分岐を持ち込まない、が既存の設計方針）。
- **2 点評価による定数畳み込みの可換性検査の恒真化線**: 第 2 点が Tmax と別の値かつ 2 以上
  でなければ恒真化する（`Dim(min=2)` 運用と 0/1 特殊化の回避線）。flow/dec で sym_max を
  4096 に上げるときもこの門は生きている。

## 5. 依存・資産

- **モデルコード**: PyPI パッケージ `style-bert-vits2==2.5.0`（PyPI registry 経由、vendor/
  clone ではない）。プロトタイプ側の exporter pyproject の `[dependency-groups].sbv2` に
  `huggingface-hub>=1.26.0` と共に列挙され、uv.lock でも registry ソースと確認されている。
  実コードはプロトタイプ側で変更せず、torch.export 可能化のための monkeypatch 層のみを
  エクスポータ側に追加する方針。テキスト層の BERT トークナイザは HuggingFace Hub リポジトリ
  `ku-nlp/deberta-v2-large-japanese-char-wwm`（DeBERTa front recon の対象と同一）を
  `huggingface-hub` 経由でロードする想定だが、ダウンロード先キャッシュや認証要否まではコード
  から未確認（推測混じり）。
- **重み**: 実重みはリポジトリに含まれず、ローカルディレクトリへ事前配置される運用。
  プロトタイプ側では `apps/playground/public/models/sbv2/` に
  `jvnv-F1-jp_e160_s14000.safetensors`（251,150,980 bytes）/ `config.json` /
  `style_vectors.npy` の実在が確認されている（`ls -la` 実測、2026-07-31 更新）。当該ディレ
  クトリは `.gitignore` で除外されており Git には一切コミットされない — 手動配置または別途
  取得スクリプトでの入手が前提。**取得元 URL・入手手順自体はコードベースからは特定できず
  未確認**（推測: HuggingFace 上の Style-Bert-VITS2 系配布モデルの可能性が高いが根拠なし）。
  ロード経路は `model_dir.glob("*.safetensors")` で単一ファイルを検出し、
  `HyperParameters.load_from_json(model_dir/"config.json")` と
  `get_net_g(str(ckpts[0]), hps.version, "cpu", hps)`（style_bert_vits2 パッケージの関数）
  でロードする形。
- **Python 依存**: `style-bert-vits2==2.5.0`（uv dependency-group）、
  `huggingface-hub>=1.26.0`（BERT トークナイザ取得用）、`safetensors>=0.8.0` /
  `torch>=2.13.0`（Karume 側と共通・既存）、`transformers>=5.14.1`（Karume 側
  pyproject には未記載 — style-bert-vits2 が内部で transformers/tokenizers に依存する可能性
  が高いが、推移的依存の全量は未精査）。
- **Karume 側との差分**: 現行 `tools/exporter/pyproject.toml` は numpy/safetensors/torch
  のみの base deps + dev group（pytest/ruff）しかない。SBV2 対応には uv の追加 dependency-
  group（名前は独自に決める必要あり）として `style-bert-vits2==2.5.0` と
  `huggingface-hub>=1.26.0` を追加する形が最小差分。transformers が SBV2 専用に必要かは
  未確認（DeBERTa export 等 SBV2 以外でも使われている可能性があり切り分けが要る）。
- **未特定事項**: 重み入手手順（配布元 URL、ダウンロードコマンド等）はコードベース内に
  見当たらず、README 等の追加ドキュメント依存の可能性がある。今回の調査では該当ドキュメント
  は未読（範囲外）。

## 6. 未検証事項

JSON 記録内で `unknown` / 推測と明記された項目、および記録自身が「未実測」「未実施」と
断っている項目を列挙する。

- **flow の matmul rank-4 が bmm へ落ちるか**: Karume の matmul は rank-2 のみ・bmm は
  rank-3 のみ（`src/ops.ts:954-990`）。DeBERTa front では core 分解が aten.matmul を
  view→bmm→view に落とし rank-3 bmm として通ることを実測済みだが、flow の scores/output/
  rel_local（4D×4D）が同じ経路になるかは**未実測**。b=1・h=2 は静的なので reshape で潰せる
  見通しはあるが検証はこれから。
- **`eq_scalar` → `bitwise_not(cast(bool))` 正規化の実グラフ検証**: flow の
  `scores.masked_fill(attn_mask == 0, -1e4)` を「新 op 追加ではなく normalize パスで
  `bitwise_not(cast(attn_mask,'bool'))` に書き換えて済ませる」という道筋は、Karume の
  cast 規約（`x → bool は x != 0`、`src/ops.ts:283`）と bitwise_not が bool 専業（`src/ops.ts:248`）
  であることから**規約上の帰結として導けるが、実グラフでの同値検証は未実施**。
- **remove_weight_norm の全ケース同値**: ビット一致は 1 ケース（z=(1,192,50)）の実測のみで
  スペック保証ではない。export の verify() で全ケースについて weight_norm 有効の参照と突合
  して確定させる必要がある。
- **enc_p の expand f32 解禁のカーネル影響**: strided コピー族は dtype パラメトリックなので
  dtype 解禁のみで足りる見込みだが、これは推測であり実装・検証はこれから。
- **flow の相対位置窓幅表とホスト側鏡像実装の式レベル一致**: Python 側の表構築関数と TS
  側生成関数が式レベルで一致することは裁定の前提だが、突合検証（golden での表入力生成を
  含む）はまだ行われていない。
- **24 層フル / 6 モジュール通しの実測**: front・flow・dec とも 1〜3 層または代表ケースでの
  op 集合確認に留まり、フルサイズでのノード数・メモリ・数値許容差の実測は未実施（DeBERTa
  front recon 側で同種の限定が明記されているのと同じ位置づけ）。
- **transformers / style-bert-vits2 のバージョン起因のグラフ変化**: モデリングコードの
  マイナー更新でグラフ形（バケット表の構成順序など）が変わりうる点は前提として記録されて
  いるのみで、SBV2 chain 側でのバージョン固定・再実測はこれから行う。
- **重みの入手手順**: §5 のとおり、配布元・取得コマンドはコードベースから特定できておらず、
  波1以降で実際に export を回す前に確認が必要な未決事項として残る。

## 7. 出典

本書は以下の先行記録の読み取りに基づく（プロトタイプ側の記録はいずれも Karume リポジトリ
外のプロトタイプ側 research/decisions/limitations/レビュー記録および patch_sbv2.py・
export_sbv2.py・export_sbv2_text.py・convert.py・normalize.py、style_bert_vits2 パッケージ
の `models_jp_extra.py` / `modules.py` / `attentions.py`、torch の decomposition 実装）。
Karume 側は `src/ops.ts` / `src/format/dims.ts` / `src/format/ir.ts` /
`src/codegen/strided.ts` / `src/runtime/plan.ts` / `src/runtime/executor.ts` /
`src/kernels/gather.ts` / `docs/decisions/0011-layout-strategy.md` /
`docs/decisions/0012-attrs-and-fused-ops.md` /
`tools/exporter/karume/{convert,normalize,ops}.py` を突合対象として読んだ。

---

**裁定は今後発行される ADR を正とする** — 本書は判断材料のスナップショットであり、実測は
M1-P3 波1以降で行う。
