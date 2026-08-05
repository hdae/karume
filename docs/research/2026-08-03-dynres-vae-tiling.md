# 動的解像度 + VAE tiling 実現可能性 recon（2026-08-03）

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。

> NOTE（同日追記）: 本文が引用する `MATMUL_KEY` / `BMM_KEY`（定数キー）は GEMM 置換
> （[decisions/0022](../decisions/0022-gemm-register-blocking.md)・60620c2）で
> `matmulKey(v4)` / `bmmKey(v4)` になり、**キーに形状由来の v4 フラグ 1 ビットが載る**。
> 決定性は不変（形状 → 1 ビットの写像）で、§3 の結論「ランタイム改修ゼロで symbolic を
> 受けられる」は変わらない — 記号次元の値により v4/scalar の 2 変種が実行時に切り替わる
> （両変種とも常設・再コンパイルはプロセス内キャッシュの初回のみ）。

読むだけの調査。コード変更・emit・GPU 実行・モデルの新規ダウンロードはしていない。
【事実】= ファイル/実測出力で確認したもの、【推測】= そこからの演繹・未実証の見込み。

---

## 0. 結論の要約

| 論点                                       | 判定                                                          | 一行                                                                                                                                                                                        |
| ------------------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ① symbolic H/W export                      | **uncertain**（形を変えれば holds / 素直な H,W では refuted） | DiT は「patchify をホストへ出して 1 シンボル S（トークン長）」なら現実的。VAE は attention の `H·W` 平坦化が次元言語（1 次元 1 シンボルの一次式）に載らず、素の H/W symbolic は構造的に不可 |
| ② ランタイムが空間 symbolic を受けられるか | **holds**                                                     | plan は run ごとに実 shape から組み直し、全カーネルの shape は uniform 経由・パイプラインキーに shape が入らない。ランタイム層の改修はゼロ                                                  |
| ③ 固定タイル graph + ホストブレンド        | **holds**                                                     | 512 用 VAE graph と 1024 用は **ノード列・重み・initializer が完全一致で shape 宣言だけが違う**（実測）。タイル用 graph は既存 512 資産そのままで、必要なのはホストグルーだけ               |

---

## 1. `--resolution` が焼き込むもの（export 側）

### 1.1 CLI と焼き込み点

- `tools/exporter/export_anima.py:548` — `--resolution`（既定 `RESOLUTION = 512`,
  `export_anima.py:125`）、`SPATIAL_COMPRESSION = 8`（`:127`）。
- 焼き込み点は **2 箇所だけ**:
  - `export_anima.py:270` `latent = args.resolution // SPATIAL_COMPRESSION` →
    `:276` `patch_anima.AnimaDit(model, latent, latent)`、`:283` の例示入力
    `randn(1, in_channels, latent, latent)`。
  - `export_anima.py:319` `latent = ...` → `:322` VAE の例示入力
    `randn(1, z_dim, latent, latent)`。
- `dynamic_shapes` は DiT / VAE とも **`None`**（`export_anima.py:301` と `:334`）＝
  完全静的グラフ。text 系だけが `Dim("T"/"Tsrc"/"Ttgt", min=2, max=sym_max)` を持つ
  （`:208`, `:246-247`）。モジュール docstring `export_anima.py:14-15` にも
  「**解像度固定の静的グラフ**」と明記。

### 1.2 実 IR で「解像度依存」なのは何か（実測）

`models/anima-f16/transformer/model.safetensors`（512px）と
`models/anima-f16-1024/…`（1024px）の IR を読み出して比較した。

**DiT（transformer）— 解像度依存の定数は 3 本だけ**【事実】:

| initializer              | shape (512px)    | 正体                                                         |
| ------------------------ | ---------------- | ------------------------------------------------------------ |
| `c_padding_channel`      | `[1,1,64,64]`    | `patch_anima.py:395` の `torch.zeros(1,1,latent_h,latent_w)` |
| `const_803f72fc05010edb` | `[1,1,1024,128]` | rope の cos 表（畳み込み済み）                               |
| `const_1b2959c9ce9c0439` | `[1,1,1024,128]` | rope の sin 表（同上）                                       |

他の 3901 ノードぶんの重みは全て解像度非依存（`[2048,2048]` 等）。
`1024` = `(64/2)·(64/2)` = patchify 後のトークン長。

**VAE decoder — 解像度依存の定数はゼロ**【事実】。512 と 1024 の
`model.safetensors` を突合すると:

```
tensor keys identical: true 108
tensor entries identical: true      # dtype/offset/shape まで完全一致 = 重みバイトも同一
nodes identical: true 455 455       # ノード列（op/ins/outs/attrs）が完全一致
initializers identical: true
values differing: 455 / 563         # 違うのは中間値の宣言 shape だけ
```

ファイルサイズ差も 50,732,956B vs 50,733,492B = **536 バイト**（IR JSON の
shape 文字列ぶん）。**VAE decoder のグラフは解像度に対して構造的に不変**である。

### 1.3 解像度依存の実体（モデル側）

- **rope**: `diffusers/models/transformers/transformer_cosmos.py:480-518`
  `CosmosRotaryPosEmbed.forward`。`pe_size = [F/pt, H/ph, W/pw]`（`:482`）から
  `emb_h` / `emb_w` を `repeat` で外積し（`:503-504`）、
  `flatten(0,2)`（`:515`）で `(F'·H'·W', dim)` に潰す。
  → **表の行 index は `h·W' + w`**。Hmax/Wmax で焼いた表の先頭を切り出しても
  別の H'/W' の表にはならない（ADR 0010 の `sym_prefix_slice` では代用不可）【事実】。
- **patchify**: `patch_anima.py:410-421`。`reshape(B,C,F/pt,pt,H/ph,ph,W/pw,pw)` →
  `permute` → `flatten(4,7).flatten(1,3)` でトークン軸 `F'·H'·W'` を作る。
  **1 次元に H と W の積**が出る。
- **unpatchify**: `patch_anima.py:440-447`（逆順）。同じく積が出る。
- **padding_mask**: `patch_anima.py:395` で latent 解像度のゼロ定数チャネルに畳んである。
  ゼロ定数なので実行時に host で作れる（グラフに焼く必然性はもう無い）。
- **VAE の attention**: `patch_anima.py:216-226` `_attention_block_forward`。
  `qkv.reshape(batch, 1, channels*3, -1)` の `-1` が **`H·W`**。実 IR では
  512px で `bmm [1,4096,384]×[1,384,4096] → [1,4096,4096]`（node 52）、
  1024px で `[1,16384,16384]`（同 node 52）【事実】。
- VAE の conv2d は **全 stride `[1,1]`**（実測: 全 conv2d ノードの attrs）。
  upsample は reshape/expand（`patch_anima.py:188-213`）で、`2H` / `2W` の一次式に載る。

---

## 2. ① export 側: symbolic H/W は可能か

### 2.1 越えられない壁 — 次元言語

正本は `docs/ir-v1.md:135-146` と `src/format/dims.ts:5-10` /
`tools/exporter/karume/dims.py:27-33`:

> 次元式の正準文法: **`coeff·sym + offset` の一次式・1 次元 1 シンボル**

エクスポータ側の門は `karume/convert.py:381-403` `_sym_parts`:
`len(symbols) != 1` なら `1 次元に複数シンボルが混ざる` で fail loudly（`:388-392`）。
係数・オフセットも整数一次式でなければ拒否（`:393-402`）。

したがって **`H·W`（2 シンボルの積）も `S²` も宣言できない**【事実】。
これは DiT の patchify（`H'·W'` トークン軸）と VAE の attention（`H·W` 系列軸）の
両方を直撃する。

補足: `shapes.py:_numel_key`（`:151-172`）は reshape の**要素数一致判定**だけは
記号の積を因数分解して扱えるが、これは「鍵の一致」であって「1 次元に積を書ける」
こととは別。個々の次元は `Extent(coeff, sym, offset)`（`shapes.py:62-83`）どまり。

### 2.2 DiT: **トークン長 1 シンボル S** なら現実的（推奨案）

グラフの入口を patchify 後にずらす。

| 現状（`patch_anima.AnimaDit.forward`）                | 提案                                                        |
| ----------------------------------------------------- | ----------------------------------------------------------- |
| 入力 `latents [1,16,H,W]`                             | 入力 `tokens [1,S,68]`（= patchify 済み。`68 = 17·2·2`）    |
| `cat` で padding channel（`:405`）                    | ホストで連結（恒常ゼロなので実質パディング）                |
| `model.rope(...)`（`:408`、cos/sin が定数に畳まれる） | 入力 `rope_cos [1,1,S,128]` / `rope_sin [1,1,S,128]` へ昇格 |
| patchify の reshape/permute 連鎖（`:410-421`）        | ホスト側（TS）で実施                                        |
| unpatchify（`:440-447`）                              | ホスト側で実施                                              |

こうすると **グラフ内に H/W が 1 つも現れない**。残るのは:

- `patch_embed.proj`（linear、最終次元 68 → 2048）
- 28 ブロックの self-attn（`bmm [16,S,128]×[16,128,S] → [16,S,S]`、softmax 最終次元 S）
- cross-attn（`encoder_hidden_states` は 512 行固定 — 解像度非依存。
  `examples/anima/README.md:45` にも明記）
- `norm_out` / `proj_out`

全て **1 次元 1 シンボル**で書ける【推測（強）】。根拠:

- レイアウト op（`slice`/`cat`/`flip`）は **宣言レベルで静的軸のみ**受理
  （`src/runtime/plan.ts:165-186` / `shapes.py:385-`）。
  実 IR の DiT にある slice は `dim:1` の `[1,6144]`（timestep 変調）と
  `dim:3` の head_dim 分割だけ、cat は `dim:3`（rope の半回転）と
  **`dim:1` の padding channel 1 本のみ**（実測。この 1 本はホストへ出る）。
  → **S 軸に触るレイアウト op は 1 つも無い**【事実】。
- 前例が 2 つある: DeBERTa（`symbols ["T"]`、入力 `[1,"T"]`、
  bmm/softmax/conv1d/sym_prefix_slice を含む 1230 ノード）と SBV2 front
  （`symbols ["P"]`、911 ノード）。どちらも実 GPU で数値一致済み
  （`.claude/ACTIVE_DESIGN.md` の M1-P2 / M1-P3 節）【事実】。
- SBV2 では「(T,T) の相対位置注意表をグラフ入力へ昇格」という**まったく同型の手**を
  既に採っている（ADR 0013 / ACTIVE_DESIGN 波7 節）。rope 表の入力昇格はその再演。

**コスト**【事実 + 推測】:

- rope 表 2 本のアップロード = `S·128·4B ×2`。1024px（S=4096）で 4MB/step。
  現状の per-step が 26.5s（`examples/anima/README.md:145`）なので無視できる。
- patchify/unpatchify のホスト計算は `17·H·W` 要素の並べ替え
  （1024px で 17·128·128 = 278K 要素）。latent は既に毎 step ホストへ戻っている
  （scheduler / CFG / Euler がホスト — `examples/anima/main.ts`）ので**追加の
  同期は発生しない**。

**障害物（残るリスク）**【推測】:

1. `torch.export` が S 依存の guard を作らずに通るか未実証（実行禁止のため未検証）。
   `CosmosTransformerBlock` 内で `.shape` から作る reshape はトークン軸を跨がない
   （head 分割は最終次元）ので通る見込み。
2. `AnimaDit` の逐語ラッパを書き換えるので、`--verify` の eager 同値（現在ビット一致）を
   採り直す必要がある。ホストへ出した patchify/unpatchify/rope の TS 実装は
   **フィクスチャとのビット一致**で固定する（SBV2 relattn パリティテストと同じ形）。
3. `min_sequence_length = 512` の `encoder_hidden_states` は解像度非依存なので影響なし。
4. `Dim("S", min=…, max=Smax)` の上限が要る（ADR 0010 の畳み込み要件）。ただし
   rope を入力へ出せば S 依存の定数は残らないはずで、`sym_prefix_slice` すら不要。

### 2.3 VAE decoder: 素の H/W symbolic は **構造的に不可**

conv 本体は symbolic に載る【事実】:

- `shapes.py:_conv_length`（`:723-763`）は記号長を一次式のまま扱い、
  `stride=1` なら `coeff` そのまま。VAE の conv2d は全て stride 1 なので
  `H → H`、`padding=1/kernel=3` でも `H` のまま。
- upsample（`patch_anima.py:206-213`）の reshape 連鎖は
  `[384H, W, 1] → [384H, W, 2] → [384, H, 2W]` で、各次元が一次式に載る
  （`_numel_key` の因数分解でも一致）。
- `_l2_normalize_channels`（`:147-158`）の permute + 最終次元 sum も symbolic 可。

**唯一の壁が mid block の attention 1 本**。`[1,C,H,W] → [1,1,3C,H·W]` の平坦化が
次元言語に載らない。回避策と評価:

| 回避策                                                | 評価                                                                                                                                                                                    |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 次元言語を「多項式」へ拡張                            | ADR 0010 / dims.ts / dims.py / `dim-grammar.json` / plan.ts / shapes.py を同時に動かす大改造。束縛の一意性（`plan.ts:276-283` の「要素数からの逆算はしない」MUST）が崩れる。**非推奨**  |
| VAE を 3 グラフに割り、attention の前後でホストへ出す | 次元言語には触らずに済む。ただし `384×H·W` の f32 を 2 往復（1024px で 25MB×2）readback + attention の `H·W`² は縮まない（1024px で 1.07GB のバッファが残る）。**メモリ問題は解けない** |
| **タイル化（③）**                                     | 動的形状が要らず、メモリ問題も同時に解ける。**推奨**                                                                                                                                    |

---

## 3. ② ランタイム側: 空間 symbolic を受けられるか

**追加改修なしで受けられる**【事実】。根拠を層ごとに:

- **plan は run ごとに組み直す**: `src/runtime/executor.ts:519`
  `const plan = planGraph(graph, bindSymbols(graph, inputShapes, bindings))` が
  `run()` の中にある。束縛は入力 shape の次元位置から取る
  （`plan.ts:285-359`）。以後 `computeOutputShape` は**解決済みの数値 shape**しか
  見ない（`ops.ts:1588-` / `plan.ts:387-391`）。つまり shape 検査は
  「静的か動的か」を区別しない。
- **アリーナは run ごと**: `executor.ts:522` `new RunArena(device, …)`、
  出力バッファは `arena.allocStorage(numel(step.outputShape) * 4)`（`:672`）で
  実 shape から確保。
- **パイプラインキーに shape が入らない**:
  `MATMUL_KEY`（`matmul.ts:19`）/ `BMM_KEY`（`bmm.ts:22`）/
  `SOFTMAX_KEY`（`softmax.ts:31`）は定数文字列。
  `conv2dKey`（`conv2d.ts:50`）は `WeightStorage` のみ。
  elementwise は rank と dtype のみ（`elementwise.ts:359`、doc `:15-18` に
  「shape は params バッファで渡すので、同じ rank ならパイプラインを使い回せる」）。
  → **解像度が変わっても再コンパイルは起きない**。
- **カーネルは全て uniform 駆動**: `conv2d.ts:60-127` の `Dims` struct が
  `height_in/width_in/height_out/width_out/stride/padding/dilation/groups` を
  全部 uniform で受け、grid-stride ループで回す。shape は WGSL に焼かれていない。
- **requiredLimits は shape 非依存**: `src/gpu/device.ts:78-95` `planRequiredLimits` は
  アダプタ上限をそのまま要求する。グラフから導いていない。

**受理集合の制約（＝ IR 側の話であってランタイム改修ではない）**:

- `slice` / `cat` / `flip` の**対象軸は宣言レベルで静的**でなければならない
  （`plan.ts:149-186` `assertStaticLayoutAxis`、ADR 0014）。
  DiT の S 軸に触るレイアウト op は無い（§2.2）ので抵触しない。
- `sym_prefix_slice` の入力は Tmax 形の静的 shape（`plan.ts:123-147`）。
- initializer は数値次元のみ（`docs/ir-v1.md:123`）。
- シンボルは「少なくとも 1 つの入力 shape に係数 1・オフセット 0 の素の形で出現」
  （`docs/ir-v1.md:143-146`）。DiT-S 案は `tokens [1,S,68]` で満たす。

**したがって②の判定は holds**。ただし「空間次元 symbolic」を
「latent の H と W をそのまま 2 シンボルで宣言する」意味に取るなら、
ランタイムではなく**次元言語**が拒否する（§2.1）。ランタイム層の改修量は **S（ゼロ）**。

---

## 4. ③ VAE tiling: 固定タイル graph + ホストブレンド

### 4.1 diffusers 実装の要点

`diffusers/models/autoencoders/autoencoder_kl_qwenimage.py`:

- 既定値（`:724-733`）: `use_tiling=False`、`tile_sample_min_height/width = 256`、
  `tile_sample_stride_height/width = 192`。
- `enable_tiling`（`:744-772`）は 4 つのノブを差し替えるだけ。
- 発火条件（`_decode`, `:843-847`）:
  `width > tile_latent_min_width or height > tile_latent_min_height`
  （latent 側の閾値 = sample 側 ÷ `spatial_compression_ratio`）。
- `enable_slicing`（`use_slicing`）は**バッチ軸の分割**（`:836-839` の
  `x.split(1)`）で、空間タイルとは無関係。Anima の B=1 経路では効かない。
- `tiled_decode`（`:983-1032`）:
  - `blend_height = tile_sample_min_height - tile_sample_stride_height`（`:996`、
    既定 256−192 = **64px**。blend 幅は **sample 空間**）。
  - latent を `tile_latent_stride`（= 192/8 = 24）刻みで走査し、各タイルは
    `[i : i+tile_latent_min_height, j : j+tile_latent_min_width]`（= 32×32 latent）。
  - タイルごとに `post_quant_conv` → `decoder` を**独立に**通す（`:1014-1017`）。
  - `blend_v` / `blend_h`（`:891-905`）は線形ランプ:
    `b[y] = a[-blend+y]·(1 − y/blend) + b[y]·(y/blend)`（上/左のタイルと現タイルの重ね合わせ）。
    `blend_extent = min(a, b, blend_extent)` で端を安全側に丸める。
  - ブレンド後、各タイルを `[:stride_h, :stride_w]` に**切り詰めて**連結し、
    最後に `[:sample_height, :sample_width]` で全体を crop（`:1024-1030`）。

### 4.2 Karume で成立するか

**graph 側: 追加作業ゼロ**【事実】。§1.2 のとおり VAE decoder のグラフは
解像度に対して構造不変で、512 用資産（`models/anima-f16/vae_decoder/`、
入力 `[1,16,64,64]`）を **latent 64×64 のタイル decoder としてそのまま使える**。
重みバイトは 1024 用資産と同一なので、資産を増やす必要もない。

**ホスト側に要るもの**:

1. latent のタイル切り出し（`Float32Array` のストライドコピー）
2. タイルごとの `session.run`（既存 API、逐次）
3. 出力 `[1,3,512,512]` の線形ランプブレンド + 貼り付け
4. 端のタイル整列（下記の**diffusers からの意図的な逸脱**）

**diffusers 式をそのまま持ち込めない点（重要）**【事実→推測】:
diffusers は `range(0, height, stride)` で走査するので**最後のタイルが短くなる**
（例: latent 128、tile 64、stride 48 → i=96 のタイルは 32 行しか無い）。
Karume のタイル graph は**形が固定**なので短いタイルを食えない。
→ 走査を「最後のタイルの開始位置を `H − tile` にスナップする」形へ変える必要がある
（例: latent 128 / tile 64 / stride 32 → 開始位置 0, 32, 64 の 3 本で
`(128−64) % 32 == 0` が成立）。この制約は「`(H_latent − tile) % stride == 0`」で、
ホスト側でタイル数から stride を決めれば常に満たせる。**ブレンド式そのものは
diffusers と同じで良い**が、重なり幅がタイル位置ごとに変わらないよう
等間隔配置にするのが素直。

**効くところ（メモリ）**【事実】:
`examples/anima/README.md:145,181` の実測 —
1024px の VAE decode 中 VRAM は **6,725MiB**（512px は **1,605MiB**）で、
「1024 チェーンの下限を決めているのは DiT ではなく VAE のほう」と明記されている。
タイル化すれば 1024px の decode ピークが 512px 相当（≈1,605MiB）まで落ちる。
attention のバッファも `[1,16384,16384]` f32 = **1.07GB** →
`[1,4096,4096]` = 67MB へ。**2048px は非タイルでは原理的に不可能**
（`65536² × 4B ≈ 17GB`）なので、タイル化は「1024 の余裕」ではなく
「それ以上への唯一の道」である【推測（強）】。

**効くところ（時間）**: 未測定【推測】。1024px を 3×3 = 9 タイルで覆うと
ピクセル換算の冗長は約 2.25 倍だが、attention の二次項は 16× → 9×(1×) に落ちる。
**速くなる可能性が高いが断言できない**（現状 1024 の decode は 15.0s、
512 は 3.7s — いずれもロード込み。`examples/anima/README.md:145`）。

**品質リスク（継ぎ目）**【事実 + 推測】:

- タイルは decoder 全体（mid block の **global attention 込み**）を独立に通るので、
  attention の受容野がタイル内に閉じる。これは diffusers の `tiled_decode` も
  まったく同じ近似（`:1014-1017` はタイルごとに `self.decoder(...)` を呼ぶ）なので
  上流と同じ品質水準になる【事実】。
- ただし Karume の既存 E2E は torch との**数値一致**で回帰を張っている
  （tolerance は実測導出）。タイル decode は torch 側も
  `vae.enable_tiling()` を呼んだ参照を採らないと突合できない。
  **参照フィクスチャの採り直しが要る**（`tools/exporter/anima_pipeline.py:333`
  `decode_latents` にタイル経路を足す形）。
- 継ぎ目は「ブレンド幅 ≥ decoder の実効受容野」であるほど目立たない。
  Anima の decoder は 3×3 conv を 12 本 + upsample ×3 なので、
  sample 空間の受容野は数十 px オーダー【推測】。diffusers 既定の
  blend 64px は同オーダーで、タイル 512px に対しては 128px 級の重なりを
  取れる（stride を絞れば良い）ので余裕がある。

### 4.3 「VAE を symbolic にする」との比較

タイル化は **① を待たずに単独で入れられる**。
①（DiT の S 化）と③（VAE タイル）は直交で、
両方入れて初めて「1 資産で任意解像度」になる（DiT 側が S 化されていなければ
DiT は依然 512/1024 別資産）。

---

## 5. 推奨の並び（工数感つき）

| 案                     | 内容                                                      | 工数          | 備考                                                                                                                             |
| ---------------------- | --------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **③ VAE タイル**       | ホストグルーのみ。graph は既存 512 資産                   | **S〜M**      | 参照フィクスチャの採り直し（torch 側 `enable_tiling` 相当）が主コスト。1024 の VAE ピーク 6,725→≈1,605MiB                        |
| **① DiT の S 化**      | patchify/unpatchify/rope をホストへ、`Dim("S")` で export | **M**         | ラッパ書き換え + TS ホスト実装 + `--verify` / golden / tolerance 採り直し。前例（DeBERTa T / SBV2 P / relattn 表の入力昇格）あり |
| ①' VAE の H/W symbolic | 次元言語を多項式へ拡張 or VAE を 3 分割                   | **L**         | 前者は ADR 0010 と束縛の一意性を壊す。後者はメモリ問題を解かない。**非推奨**                                                     |
| ② ランタイム改修       | 不要                                                      | **—（ゼロ）** | plan は run ごと・キーに shape なし・limits は shape 非依存                                                                      |

---

## 6. 参照した file:line 一覧（主要）

- `tools/exporter/export_anima.py:14-15,125-127,208,246-247,270-307,313-339,548`
- `tools/exporter/karume/patch_anima.py:147-158,188-213,216-226,370-447`
- `tools/exporter/karume/dims.py:27-33,55-71`
- `tools/exporter/karume/convert.py:381-403`
- `tools/exporter/karume/shapes.py:62-83,105-172,333-345,385-,700-763`
- `src/format/dims.ts:5-10,19-21,39-56`
- `src/runtime/plan.ts:52-57,111-186,269-359,361-415`
- `src/runtime/executor.ts:519-569,627-680`
- `src/ops.ts:1588-1800`（`computeOutputShape`）
- `src/kernels/conv2d.ts:50-137`、`src/kernels/bmm.ts:22`、
  `src/kernels/matmul.ts:19`、`src/kernels/softmax.ts:31`
- `src/codegen/elementwise.ts:1-20,359`
- `src/gpu/device.ts:78-95`
- `docs/ir-v1.md:123,135-146`
- `.claude/ACTIVE_DESIGN.md`（M1-P2 / M1-P3 / M1-P4 / 画像デモ節）
- `examples/anima/README.md:45,62,85-90,142-181`
- `.venv/…/diffusers/models/transformers/transformer_cosmos.py:457-518`
- `.venv/…/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py:715-772,836-847,891-905,983-1032`
