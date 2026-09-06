# 融合スパイク K-15 / K-7 の ABBA 実測と P-5 の計測 2 段・fusion-hints 再掃引（2026-09-06）

> 時点スナップショット（RTX 3080 Ti / Vulkan・Deno 2.9.6）。事前ベースラインは
> [2026-09-06-op-fusion-baseline](2026-09-06-op-fusion-baseline.md)、候補の採否は
> [perf-ledger](../perf-ledger.md)。スパイク実装 = `4dd5e1f`（K-15 `geluTanhMul`）/ `8c80aab`
> （K-7 `gatedResidual`）。生データは `outputs/bench/<資産>/2026-09-06_ab-fusion-*`・
> `2026-09-06_single-strided`・`2026-09-06_fusion-hints`（git 追跡外）。

## §0 要約

- **K-15 / K-7 とも判定線に届かない**（§4）。gemma4 decode は GPU −0.16〜−0.33 ms/token（−0.8〜−1.5%・
  うち半分は K-7 側が掴んだ RoPE 末尾）で壁は best-vs-best −0.4%（線 = 壁 −1%/token）。anima は
  transformer step −7.1〜−7.9 ms（−0.9%/step・全 GPU −0.6%）、irodori DiT は −0.09 ms/step（−0.2%）
  （線 = 全 GPU −1%）。両ルールともビット一致（PNG / wav / golden 門緑）で、効果は帯域算術どおり
  「中間 1 本の往復ぶん」だけ — dispatch 数の削減そのものは GPU 時間にも壁にも出ない。
- **P-5 は実装せず保留**（§1）: 消費側 3 系統（rms_norm 50% / linear 32% / attention 18%）の改造が要り、
  `permute` 単体は既に実効 691 GB/s（理論の 76%）。消えうる最大 1.5〜1.8% は判定線 −1.3% と同じ桁で、
  消費側の追加費用で消える。
- **fusion-hints の再掃引は 09-04 と同一**（§2）。新しい候補は無い。
- 含意: 要素ごと op の融合（dispatch ダイエット）は 4 家族とも 1% 以下の帯で尽きた。残る時間は
  linear（57〜87%）・attention・anima VAE（§5）。

## §1 P-5（`permute` の消費側畳み込み）— 計測 2 段

1. **消費側の実数**（anima transformer・census 2026-09-04・素の `permute` 224 本 / 1,468.0M 要素）:
   読み手は `rms_norm` 112 本（734M 要素・50%）/ `reshape` 越しの `linear` 56 本（470M・32%）/
   `attention` 56 本（264M・18%）。作り手は `reshape`（linear の後）168 本 / `attention` 56 本。
2. **単体の実効帯域**（`opbench single --op permute --component transformer`）: 加重 16.98 ms/step
   （3 形・224 本）。読み書き 11.74 GB ÷ 16.98 ms = **691 GB/s**（実グラフ内の strided 族 20.7 ms は
   slice 0.93 ms を含み、single / graph ≈ 0.86）。
   ```bash
   deno run -A tools/opbench/main.ts single --census outputs/bench/karume-anima/2026-09-04_op-census --scenario 1024px --component transformer --op permute --op slice --op cat --op expand --out outputs/bench/karume-anima/2026-09-06_single-strided
   ```
3. 判定: 畳んで消えるのは permute カーネルの書き 1 回 + 消費側の読み 1 回 = 最大 17〜20 ms/step
   = 4 step で 68〜80 ms = **全 GPU の 1.5〜1.8%**（判定線 −1.3% = 59 ms）。消費側が strided で読む
   費用（GEMM のタイル充填の連続性・attention の K/V 読み）を引くと届かない。**実装せず保留**
   （2026-09-06 ユーザー裁定 — 他の余地が尽きたら試す優先度）。

## §2 fusion-hints 再掃引（HEAD `a073294`・窓幅 12・11 資産）

09-04 の掃引と候補行は同一（9 資産で完全一致）。差は 2 つで、gemma4 はミラー改名 `karume-gemma4-e2b`
→ `karume-gemma4` の `source` 欄だけ（候補 209 行同一）、birefnet-hr-1024 は decoder パッチ ⑨（1×1 conv →
upsample の順）で 717 → 721 行（`cat,conv2d` 11 → 10・`add,conv2d` 系 2 行が増減）。
窓幅 12 の主な極大鎖（別名化だけの鎖を除く・本数順）:

| 家族 / グラフ             | 極大鎖（本数）                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| anima transformer         | `linear,mul,add` 84（→ K-7 で消化）/ `slice,reshape` 254 / `linear,gelu,linear,mul,add` 28                    |
| anima text_encoder        | `linear,reshape,rms_norm,permute` 56                                                                          |
| anima vae_decoder         | `clamp_min,reshape,div,mul,mul` 30（weight-norm 相当）                                                        |
| gemma4 decode             | `linear,rms_norm,add` 105 / `rms_norm,add` 106（残差 + 次段 norm — normalize の出自で Inductor と join 不可） |
| irodori dit               | `linear,reshape` 109 / `rms_norm,mul` 60 / `mul,add` 72（うち 24 = K-7・24 = adaLN 変調・24 = 偶奇 RoPE）     |
| irodori backbone          | `slice,reshape,permute` 75 / `layer_norm,linear` 49                                                           |
| irodori codec_*           | `mul,sin` 29 ×2（Snake 活性）/ `add,reshape,conv1d,reshape` 12〜13                                            |
| sbv2 text_encoder / voice | `linear,reshape,permute,reshape` 110 / `expand,reshape,bmm,reshape` 96                                        |
| birefnet                  | `add,softmax,expand,reshape,expand,reshape,bmm,reshape,permute,…` 48（分解 attention・マスク加算入り）        |
| siglip2 so400m / base     | `linear,add` 55 / 25・分解 attention の完全一致形 28 / 13                                                     |
| depth-anything            | `linear,mul,add` 24 / 分解 attention 12                                                                       |

## §3 スパイク実装（受理集合と実資産のヒット）

- **K-15 `geluTanhMul`**（`4dd5e1f`）: `gelu_tanh(g) → mul(·, u)` の隣接 2 ノード・同 shape f32・中間 private。
  gemma4 decode で **35 対**（per-layer 入力ゲート `[1,M,256]`）。**MLP の 35 対は掴めない** — 発行順が
  `linear(gate) → gelu_tanh → linear(up) → mul` で、up 射影が間に入る（跨ぐには passthrough が要る）。
  slot 順は 70 本すべて gelu が slot 0（両順を受理してキーに残したが実資産に `u-gelu` は無い）。
  EmbeddingGemma / MiniCPM5 / anima / irodori は 0。
- **K-7 `gatedResidual`**（`8c80aab`）: `mul(gate[1,…,1,dim], x) → add(residual, ·)`・残差は add の slot 0
  固定・gate だけ broadcast。anima transformer **84 / step**・irodori dit **24** + duration 3・gemma4 decode
  **35**（rope が掴み損ねた 35 鎖の末尾 `mul(x, sin) → add` — 値は同一）。irodori の rms_norm ベース
  adaLN 変調（`add,mul,add` 24・両方 broadcast）と偶奇 RoPE は受理集合の外（fusion-hints の門で固定）。
- 両方とも silu.ts の手筋（中間を workgroup メモリへ書き barrier 後に読み戻す）で素の 2 dispatch と
  ビット一致（実 GPU A/B テスト。K-7 は障壁を外すと 2 ULP 差で落ちることをフォールト注入で確認）。
  anima PNG sha256 門 / irodori wav・latent 門 / gemma4 golden 厳密一致とも緑。
- **訂正**: 事前の census 読み（[baseline](2026-09-06-op-fusion-baseline.md) §3 の K-15 行）は
  「隣接 70 対・slot 順 2 通り」としていたが、隣接は 35 対・slot 順は 1 通り。`consumers` は隣接を
  意味せず、`producers` の並びは ins の slot 順ではない — 形の確定は IR 本体で行う（K-7 では第 0 段
  として IR から数えた）。

## §4 ABBA 実測（A = `a073294` の worktree・B = `8c80aab`・A B B A の順・同一セッション）

timing は `graph --census` の op 別 GPU 時間、wall は別プロセス。gemma4 の wall は greedy が 29 token で
EOS に達するため prefill + 29 token の合計（両側同じ列）。anima は 4 run 中に単調なドリフト（A1 810 →
B1 816 → B2 891 → A2 927 ms/step）が乗っており、隣り合う対で読む。

| 家族 / 量                        |          A（前） |          B（後） |                                             差 |
| -------------------------------- | ---------------: | ---------------: | ---------------------------------------------: |
| gemma4 decode GPU / token        | 20.31 / 21.98 ms | 21.25 / 20.59 ms |                                       ノイズ内 |
| 〃 add + mul + gelu_tanh + fused | 1.357 / 1.489 ms | 1.193 / 1.164 ms |            **−0.16 / −0.33 ms**（−0.8〜−1.5%） |
| 〃 dispatch / token              |            1,373 |            1,303 |       −70（geluTanhMul 35 + gatedResidual 35） |
| 〃 壁（prefill + 29 token）      | 3,230 / 3,704 ms | 3,218 / 3,237 ms |                         best-vs-best **−0.4%** |
| anima transformer GPU / step     | 810.6 / 926.9 ms | 816.3 / 891.4 ms |                 対で +0.7% / −3.8%（ドリフト） |
| 〃 add + mul + gelu + fused      | 46.78 / 48.80 ms | 39.69 / 40.91 ms | **−7.1 / −7.9 ms**（−0.9%/step・全 GPU −0.6%） |
| 〃 dispatch / step               |            2,316 |            2,232 |                                            −84 |
| 〃 壁（4 step）                  | 7,937 / 8,683 ms | 7,958 / 8,148 ms |                 対で +0.3% / −6.2%（ドリフト） |
| irodori DiT GPU / step           | 38.39 / 38.07 ms | 38.07 / 38.08 ms |                                      **−0.2%** |
| 〃 add + mul + fused             | 2.370 / 2.358 ms | 2.269 / 2.274 ms |                                       −0.09 ms |
| 〃 dispatch / step               |            1,596 |            1,572 |                                            −24 |
| 〃 壁（発話 1 本）               | 3,555 / 3,859 ms | 3,550 / 3,560 ms |                             best-vs-best −0.1% |

読み: anima の −7 ms/step は「消えた mul 7.3 + add 10.2 = 17.5 ms」と「増えた融合カーネル 10.4 ms」の差で、
= 中間 `[1,4096,2048]` f32 33.5 MB の書き + 読み × 84 本 = 5.6 GB ÷ ≈800 GB/s。帯域算術そのもので、
dispatch 84 本ぶんの起動費は見えない。gemma4（M=1・launch 律速）でも 70 dispatch の削減は
0.16〜0.33 ms/token で、壁 32 ms/token の 1% に届かない。

**判定**: K-15 = kill（壁 −1%/token に届かない — 掴めていない MLP 側 35 対を足しても最大 2 倍）。
K-7 = kill（全 GPU −1% に届かない — anima −0.6% / irodori −0.2%）。

## §5 含意

1. 要素ごと op の融合は、この 4 家族では効果が「中間 1 本の往復」に限られ、どれも全 GPU の 1% 以下。
   dispatch 本数の削減は M=1 の decode でも壁に出ない（フェンス床 ≈10 ms/token — H-2 — が支配）。
2. 残る時間は linear（gemma4 decode 69%・anima step 57%・siglip2 79〜87%・irodori DiT 63%）、attention
   （anima 27%/step・siglip2 so400m の分解 attention 17%）、anima VAE decoder（全 GPU の 29%）。
   gemma4 の linear は GEMV の並列度（K-14 / K-16）、siglip2 / depth / birefnet の分解 attention は
   行ブロック attention の受理を `softmax` 綴りへ広げる案（[op-census](2026-09-03-op-census-fusion-hints.md)
   §2.3 の 128 本）が次の候補。
3. 道具: `opbench graph` の A/B は同一セッションでも anima 4 run で 10% 級のドリフトが乗る。長い run
   は A B B A の隣接対で読むか、round を増やして min を採る規約が要る（baseline §4 の宿題に追加）。
