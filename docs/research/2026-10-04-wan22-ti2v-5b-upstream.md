# Wan2.2 TI2V-5B の上流構造 — Wan2.1 T2V 1.3B との差（2026-10-04 時点）

> 時点スナップショット（2026-10-04）— Wan2.2 TI2V-5B を Karume へ移植する ADR の材料として、上流の構造・数値を一次ソースから写した調査記録（裁定は含まない）。

手法: Wan-Video/Wan2.2 を clone して読み、HF の 2 リポ（公式形式 / Diffusers 形式）を HF API と HTTP Range で調べた
（safetensors はヘッダと一部テンソルの先頭だけを読んだ。重み本体は落としていない）。diffusers は
`tools/.venv` に入っている 0.39.0（Karume の wan recipe が pin している版 — ADR 0118 決定 7）を読み、VAE の形と
conv の MAC は diffusers の `AutoencoderKLWan` を meta device で組んで実際に forward させて取った。GPU は使っていない。

記法:

- 「一次」= 開発元のソース・配布物・公式ドキュメント。「二次」= それ以外。推測には（推測）を付ける。
- 「観測」= こちらで実行して得た値（コマンドは本文か §11 に記す）。
- `Wan2.2/...:N` は GitHub Wan-Video/Wan2.2 の commit `1ea34ff48f87168174e12956e200b1d908b1c5ff`
  （2026-09-21）。`transformer_wan.py` などの diffusers の `file:line` は diffusers 0.39.0 の
  `tools/.venv/lib/python3.14/site-packages/diffusers/` 配下。
- HF の pin: 公式形式 `Wan-AI/Wan2.2-TI2V-5B` @ `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`、
  Diffusers 形式 `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff7315c768468a5333511427288870b2e9635`。
- 比較対象 Wan2.1 1.3B の値は [2026-10-02 の調査](2026-10-02-video-gen-recon.md)（以下「2.1 調査」）と
  ADR [0118](../decisions/0118-wan21-video-generation.md) から引いた。

## 0. 結論の要約

- **DiT は Wan2.1 と同じクラス・同じ 825 テンソルの構成で、寸法だけが大きい**（dim 3072・24 heads・30 層・
  FFN 14336・in/out 48 ch）。パラメータは 4,999,787,712（2.1 の 3.52 倍）。重み名は 2.1 の Diffusers 版と
  同じ規則（観測）。
- **forward の差は 2 点だけ**（Wan2.1 と Wan2.2 の `wan/modules/model.py` の diff — 観測）。
  - タイムステップがトークンごと `[B, seq_len]` になり、変調 `e0` が `[B, S, 6, C]`、head の変調が `[B, S, C]` に
    なる。
  - I2V 用の CLIP 画像エンコーダと `WanI2VCrossAttention`（`k_img` / `v_img`）が無い。TI2V の I2V は「先頭の潜在
    フレームを画像の潜在で置き換え、そのトークンだけ t = 0 にする」だけで実現される。
- **VAE は別物**（Wan2.2-VAE）。z_dim 48・圧縮 4×16×16（空間 2×2 の patchify + 3 回の空間ダウン）・
  パラメータ 704,688,668（デコーダ 555,049,228 = 2.1 のデコーダの 7.6 倍）。残差の外側に `AvgDown3D` /
  `DupUp3D`（チャネルと時空間の並べ替え + 平均 / 複製）のショートカットが付く。
- **トークン数は同じ画素数なら 2.1 の 1/4**（VAE の空間圧縮が 2 倍ずつ）。既定 1280×704・121 フレームで
  27,280 トークン・1 forward 258.5 T MAC（2.1 の 832×480・81 フレームの 1.83 倍）。832×480・81 フレームなら
  8,190 トークン・49.0 T MAC で、2.1 の同条件の 0.35 倍（観測: 計算）。
- **umT5 は Wan2.1 と同じ重み**。公式形式の `models_t5_umt5-xxl-enc-bf16.pth` は 2.1 1.3B と sha256 が一致。
  Diffusers 版は 2.1 が f32・2.2 が bf16 で sha は違うが、テンソル名・形は同一で、抜き取った 79,872 値が
  「2.1 の f32 を最近接偶数丸めで bf16 にした値」とビット一致した（観測）。T5 とトークナイザのソースも
  Wan2.1 と diff 0。
- **サンプラは Wan2.1 と同じコード**（`fm_solvers*.py` が diff 0）。既定は UniPC・50 steps・shift 5.0・
  guide 5.0・24 fps・121 フレーム。Diffusers 版の `flow_shift` も 5.0。
- **export の上流は Diffusers 版が扱いやすい**。全部品が safetensors で、DiT / VAE とも Karume が既に切り出している
  diffusers のクラス（`WanTransformer3DModel` / `AutoencoderKLWan`）の分岐で表せる。DiT の値は公式形式とビット
  一致（抜き取り — 観測）。

## 1. DiT の config・パラメータ・テンソル構成

### 1.1 config（一次）

| 項目                 | Wan2.1 T2V 1.3B                     | Wan2.2 TI2V-5B                                  | 根拠（2.2）                                              |
| -------------------- | ----------------------------------- | ----------------------------------------------- | -------------------------------------------------------- |
| dim                  | 1536                                | **3072**                                        | `wan_ti2v_5B.py:21`、HF `config.json`                    |
| ffn_dim              | 8960                                | **14336**                                       | `wan_ti2v_5B.py:22`                                      |
| num_heads × head_dim | 12 × 128                            | **24 × 128**                                    | `wan_ti2v_5B.py:24`、Diffusers `attention_head_dim: 128` |
| num_layers           | 30                                  | 30                                              | `wan_ti2v_5B.py:25`                                      |
| in_dim / out_dim     | 16 / 16                             | **48 / 48**                                     | HF `config.json`（`in_dim`/`out_dim` 48）                |
| patch_size           | (1,2,2)                             | (1,2,2)                                         | `wan_ti2v_5B.py:20`                                      |
| text_len / text_dim  | 512 / 4096                          | 512 / 4096                                      | `shared_config.py:11`、Diffusers `text_dim: 4096`        |
| freq_dim             | 256                                 | 256                                             | `wan_ti2v_5B.py:23`                                      |
| eps                  | 1e-6                                | 1e-6                                            | `wan_ti2v_5B.py:29`                                      |
| qk_norm              | True（dim 全体の RMSNorm）          | True（同じ・Diffusers `rms_norm_across_heads`） | `wan_ti2v_5B.py:27`                                      |
| cross_attn_norm      | True                                | True                                            | `wan_ti2v_5B.py:28`                                      |
| window_size          | (-1,-1)                             | (-1,-1)                                         | `wan_ti2v_5B.py:26`                                      |
| model_type           | t2v                                 | **ti2v**                                        | HF `config.json`、`model.py:359`                         |
| RoPE                 | t/h/w = 44/42/42 次元・表 1024 位置 | 同じ（head_dim 128 で式が同じ）                 | `model.py:398-405`、Diffusers `rope_max_seq_len: 1024`   |
| 重みの dtype         | F32                                 | F32（bf16 由来ではない — 下記）                 | 観測                                                     |

- HF 公式形式の `config.json`（`_class_name: WanModel`・`_diffusers_version: 0.33.0`）は `dim 3072 / eps 1e-06 /
  ffn_dim 14336 / freq_dim 256 / in_dim 48 / model_type ti2v / num_heads 24 / num_layers 30 / out_dim 48 /
  text_len 512` だけを持つ。patch_size・qk_norm・cross_attn_norm・text_dim・window_size は `ignore_for_config`
  （`model.py:299-301`）で、`__init__` の既定値が使われる（2.1 と同じ事情）。
- Diffusers 版 `transformer/config.json`（`_diffusers_version: 0.35.0.dev0`）: `attention_head_dim 128 /
  cross_attn_norm true / eps 1e-06 / ffn_dim 14336 / freq_dim 256 / image_dim null / in_channels 48 /
  num_attention_heads 24 / num_layers 30 / out_channels 48 / patch_size [1,2,2] / qk_norm rms_norm_across_heads /
  rope_max_seq_len 1024 / text_dim 4096 / added_kv_proj_dim null / pos_embed_seq_len null`。`image_dim null` は
  CLIP の画像埋め込みが無いことを示す。

### 1.2 パラメータとテンソル（観測: safetensors ヘッダ集計）

`node hdr.mjs`（HTTP Range でヘッダだけを読むスクリプト）→ `node ana.mjs` で集計した（§11）。

| 量                                 | Wan2.1 1.3B   | Wan2.2 TI2V-5B                                                   |
| ---------------------------------- | ------------- | ---------------------------------------------------------------- |
| テンソル数                         | 825           | 825（公式形式・Diffusers 形式とも）                              |
| 総パラメータ                       | 1,418,996,800 | **4,999,787,712**（全 F32）                                      |
| 1 ブロック                         | 46,440,704    | 163,656,704                                                      |
| ブロック外（埋め込み・time・head） | 25,775,680    | 90,086,592                                                       |
| 2D の重み（Linear）                | 306           | 306（ブロック 30 × 10 + text 2 + time 2 + time_proj 1 + head 1） |
| Linear 重みの要素数                | —             | 4,996,792,320（全体の 99.94%）                                   |
| f16 換算                           | 2.84 GB       | 9,999,575,424 B（9.31 GiB）                                      |
| Linear だけ i8・他 f16 の換算      | —             | 約 4.66 GiB（scale を除く — 観測: 計算）                         |

- 1 ブロックの中身（Diffusers 名）: `attn1.{to_q,to_k,to_v,to_out.0}`（3072×3072 + bias）・`attn1.norm_{q,k}`
  （3072）・`attn2.*`（同じ形・k/v も 3072→3072 — 文脈は先に 3072 へ射影済み）・`ffn.net.0.proj` 14336×3072・
  `ffn.net.2` 3072×14336・`norm2`（affine・3072）・`scale_shift_table [1,6,3072]`。
- ブロック外: `patch_embedding.weight [3072,48,1,2,2]`・`condition_embedder.text_embedder.linear_{1,2}`
  （4096→3072→3072）・`condition_embedder.time_embedder.linear_{1,2}`（256→3072→3072）・
  `condition_embedder.time_proj` 3072→18432・`proj_out` 3072→192（= 48 ch × 1×2×2）・トップの
  `scale_shift_table [1,2,3072]`（公式名 `head.modulation`）。
- **重み名の規則は 2.1 と同じ**（公式 `blocks.N.norm3` ↔ Diffusers `blocks.N.norm2` のねじれも同じ — 2.1 調査
  §3.2）。観測: 公式 `blocks.7.norm3.weight` と Diffusers `blocks.7.norm2.weight` がビット一致。
- **公式形式と Diffusers 形式の DiT は同じ値**（観測 `node cmp2.mjs`: `head.modulation`↔`scale_shift_table`
  6,144 値・`blocks.0.self_attn.q.weight`↔`blocks.0.attn1.to_q.weight` 先頭 65,536 値・
  `blocks.29.ffn.2.weight` 先頭 65,536 値・`patch_embedding.weight` 先頭 65,536 値が全てビット一致）。
- **f32 の重みは bf16 の拡張ではない**（観測 `node low16.mjs`: 4 テンソルの先頭最大 65,536 値で、下位 16 ビットが
  0 の値は 0〜2 個。抜き取りの |w| の最大は 0.25 で f16 の範囲内）。

## 2. DiT の forward の差

### 2.1 Wan2.1 と Wan2.2 の `model.py` の diff（一次・観測）

`diff Wan2.1/wan/modules/model.py Wan2.2/wan/modules/model.py`（Wan2.1 は main を raw で取得）の差は次だけ。

- `WanI2VCrossAttention`（`k_img` / `v_img` / `norm_k_img`）と `WAN_CROSSATTENTION_CLASSES` の削除。cross-attn は
  `WanCrossAttention` 1 種（`model.py:158-180`）。`img_emb`（CLIP 用 MLP）も無い（`model.py:377-395` に定義が無い）。
- ブロックの `e` が `[B, 6, C]` → `[B, L1, 6, C]`、`modulation.unsqueeze(0) + e` を `dim=2` で 6 分割し、各成分を
  `squeeze(2)` して `[B, L1, C]` で掛ける（`model.py:237-255`）。式（LayerNorm → `×(1+scale)+shift` → 枝 →
  `×gate` で残差）は 2.1 と同じ。
- head の `e` が `[B, C]` → `[B, L1, C]`（`model.py:279-291`）。
- `torch.cuda.amp` → `torch.amp.autocast('cuda', ...)` の書き換え（意味は同じ）。

### 2.2 トークンごとのタイムステップの経路（一次）

- 公式 `WanModel.forward`（`model.py:459-469`）: `t` が 1 次元なら `[B, seq_len]` に expand。`t.flatten()` →
  sinusoidal 256 次元（float64・cos 先 sin 後 — `model.py:14-24`）→ `unflatten(0,(bt,seq_len))` →
  `time_embedding`（256→3072→SiLU→3072）で `e [B,S,3072]` → `time_projection`（SiLU→3072→18432）を
  `unflatten(2,(6,dim))` して `e0 [B,S,6,3072]`。どちらも f32 の autocast 区間。
- パイプラインは t2v でも i2v でも `[1, seq_len]` を渡す（`textimage2video.py:373-378`・`573-578`）。値は
  `mask2[0][0][:, ::2, ::2] * t` を flatten したもの（潜在 `[T,h,w]` を patch 格子 `[T,h/2,w/2]` に間引いた
  マスク × t）。seq_len 未満の余りは `ones * t` で埋める。
- diffusers（`transformer_wan.py:674-678`）: `timestep.ndim == 2` なら `ts_seq_len = timestep.shape[1]` として
  flatten し、`WanTimeTextImageEmbedding.forward`（`transformer_wan.py:330-351`）が `Timesteps` の出力を
  `unflatten(0, (-1, timestep_seq_len))`。`timestep_proj.unflatten(2, (6, -1))` で `[B,S,6,C]`
  （`transformer_wan.py:683-688`）。ブロックは `temb.ndim == 4` の分岐（`transformer_wan.py:469-485`・コメント
  「wan2.2 ti2v」）、head は `temb.ndim == 3` の分岐（`transformer_wan.py:704-708`）。
- `_keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]`
  （`transformer_wan.py:549`）。

### 2.3 I2V の条件づけ（一次）

- `masks_like([noise], zero=True)`（`utils.py:172-199`）は generator 無しなら `out1[:,0] = 0`・`out2[:,0] = 0`
  （先頭の潜在フレームだけ 0・他は 1）。generator 付きの確率的な分岐（p = 0.2 で `exp(N(-3.5,0.5))` を入れる）は
  学習用で、推論の呼び出しは generator を渡さない（`textimage2video.py:358`・`550`）。
- 公式 i2v（`textimage2video.py:512`・`549-551`・`597-598`）: 画像 1 枚を VAE で encode した `z [48,1,h,w]` で
  `latent = (1 - mask2) * z + mask2 * latent` と先頭フレームを置き換え、**各スケジューラ step の後にも同じ置き換え**
  をする。タイムステップのマスクで先頭フレームのトークンは t = 0（`textimage2video.py:573`）。
- diffusers `WanImageToVideoPipeline`（`expand_timesteps=True` の分岐）:
  - `prepare_latents` は画像だけを encode し（`pipeline_wan_i2v.py:425-426`）、`(latent - mean) * (1/std)` で
    正規化した `latent_condition` と `first_frame_mask`（`[1,1,T,h,w]`・先頭フレーム 0）を返す
    （`pipeline_wan_i2v.py:459-466`）。
  - 各 step で**モデル入力だけ**を `(1 - mask) * condition + mask * latents` にし、タイムステップも同じマスクで作る
    （`pipeline_wan_i2v.py:760-767`）。スケジューラの状態は置き換えない latents のまま進め、最後に 1 回置き換える
    （`pipeline_wan_i2v.py:816-817`）。
- CLIP 画像エンコーダは無い（公式 `textimage2video.py` は CLIP を import しない。Diffusers 版 `model_index.json`
  にも `image_encoder` が無い。DiT の `image_dim: null`）。

### 2.4 推測

- （推測）公式と diffusers の I2V のマスクの扱い（step 後に置き換えるか、モデル入力だけ置き換えるか）は、先頭フレーム
  以外の出力を変えない。UniPC の更新は要素ごとの演算で位置を跨がず、モデル入力はどちらも同じ値になるため。先頭
  フレームは最後に必ず画像の潜在で上書きされる。
- （推測）T2V では全トークンの t が同じなので、`[B,S,6,C]` の変調は `[B,6,C]` を S 方向へ broadcast したものと
  要素ごとに同じ値になる（同じ入力に同じ MLP を掛けるだけ）。I2V でも t の値は 2 種（先頭フレームの 0 と、それ
  以外の t）だけなので、MLP を 2 行だけ回して「先頭フレームのトークン列」と「残り」に配ればよい。既定解像度で
  トークンごとに回すと `time_projection` だけで 1 forward あたり約 1.8 T MAC、`e0` が f32 で 1.87 GiB になる
  （観測: 計算・§4.4 の表）。

## 3. VAE 2.2（Wan2.2-VAE）

### 3.1 構成値（一次）

| 項目                             | Wan2.1 VAE                         | Wan2.2 VAE                                                                            | 根拠（2.2）                                                                                    |
| -------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| z_dim                            | 16                                 | **48**                                                                                | `vae2_2.py:892`、Diffusers `z_dim: 48`                                                         |
| 圧縮（T×H×W）                    | 4×8×8                              | **4×16×16**（patchify 2 込み）                                                        | `wan_ti2v_5B.py:17`、Diffusers `scale_factor_spatial 16 / scale_factor_temporal 4`             |
| patchify                         | 無し                               | **2×2**（RGB 3 ch → 12 ch・出力も 12 → 3）                                            | `vae2_2.py:280-313`・`785`・`837`、Diffusers `patch_size 2 / in_channels 12 / out_channels 12` |
| エンコーダの base dim            | 96                                 | **160**                                                                               | `vae2_2.py:893`、Diffusers `base_dim 160`                                                      |
| デコーダの base dim              | 96                                 | **256**                                                                               | `vae2_2.py:739`（`dec_dim=256`）、Diffusers `decoder_base_dim 256`                             |
| dim_mult                         | [1,2,4,4]                          | [1,2,4,4]                                                                             | `vae2_2.py:895`                                                                                |
| num_res_blocks                   | 2（デコーダ 3）                    | 2（デコーダ 3）                                                                       | `vae2_2.py:659`（`mult=num_res_blocks+1`）                                                     |
| temperal_downsample              | [F,T,T]                            | [F,T,T]（デコーダは逆順 [T,T,F]）                                                     | `vae2_2.py:896`・`754`                                                                         |
| 残差ブロックの外のショートカット | 無し                               | **AvgDown3D / DupUp3D**                                                               | `vae2_2.py:316-497`、Diffusers `is_residual: true`                                             |
| アップサンプルの conv2d          | dim → dim/2                        | **dim → dim**（チャネルを半分にしない）                                               | `vae2_2.py:86-96`、2.1 は `vae2_1.py:75-82`                                                    |
| attention                        | mid に単一 head 1 つ               | 同じ（デコーダ D = 1024・エンコーダ D = 640）                                         | `vae2_2.py:238-277`・`546-550`・`643-647`                                                      |
| 正規化                           | RMS_norm（`F.normalize`×√C×γ）     | 同じ                                                                                  | `vae2_2.py:45-59`                                                                              |
| パラメータ                       | 126,892,531（デコーダ 73,295,331） | **704,688,668**（エンコーダ 149,627,776・デコーダ 555,049,228・quant_conv 等 11,664） | 観測（ヘッダ集計と meta での組み立ての両方）                                                   |
| ファイル                         | `Wan2.1_VAE.pth` 507,609,880 B     | `Wan2.2_VAE.pth` 2,818,839,170 B（pickle）/ Diffusers safetensors F32 2,818,777,808 B | HF API                                                                                         |

- 潜在の mean / std は 48 値ずつ（`vae2_2.py:904-1011`）。Diffusers の `latents_mean` / `latents_std` と同じ値
  （HF `vae/config.json` を目視で突き合わせた。表記の末尾 0 の有無だけが違う）。正規化は encode で
  `(mu - mean) * (1/std)`、decode で `z / (1/std) + mean`（`vae2_2.py:803-818`）。
- Wan2.2-VAE の f16 / f32 の大きさ: デコーダだけで f32 2.07 GiB / f16 1.03 GiB、エンコーダは f16 0.28 GiB
  （観測: 計算）。

### 3.2 層構成（一次 + 観測）

エンコーダ（`Encoder3d`・`vae2_2.py:500-613`）。dims = [160,160,320,640,640]。

1. patchify（`b c f (h q) (w r) -> b (c r q) f h w`）→ CausalConv3d(12→160, k3)。
2. down0: Res(160→160)×2 → downsample2d（`ZeroPad2d((0,1,0,1))` → Conv2d 3×3 stride 2）＋ AvgDown3D(160→160,
   t1, s2)。
3. down1: Res(160→320)・Res(320)×1 → downsample3d（空間 stride 2 ＋ `time_conv` CausalConv3d(320→320, (3,1,1),
   stride (2,1,1), padding 0)）＋ AvgDown3D(160→320, t2, s2)。
4. down2: 同型（320→640）。
5. down3: Res(640)×2・ダウン無し ＋ AvgDown3D(640→640, t1, s1) = **恒等の加算**（`x + x_copy` —
   `vae2_2.py:447-452` は常に足す）。
6. mid: Res(640) → Attention(640) → Res(640)。head: RMS_norm → SiLU → CausalConv3d(640→96)。
7. 外で `conv1`（CausalConv3d 96→96, k1）→ `chunk(2)` で mu を取る（`vae2_2.py:803`）。

デコーダ（`Decoder3d`・`vae2_2.py:616-723`）。dims = [1024,1024,1024,512,256]。

1. 外で `conv2`（CausalConv3d 48→48, k1 — Diffusers 名 `post_quant_conv`）。
2. conv_in: CausalConv3d(48→1024, k3)。mid: Res(1024) → Attention(1024) → Res(1024)。
3. up0: Res(1024)×3 → upsample3d（`time_conv` 1024→2048 (3,1,1) でフレーム 2 倍 → nearest-exact ×2 → Conv2d
   1024→1024）＋ DupUp3D(1024→1024, t2, s2)。
4. up1: 同型（1024 ch・時間も 2 倍）。
5. up2: Res(1024→512)・Res(512)×2 → upsample2d（nearest-exact ×2 → Conv2d 512→512）＋ DupUp3D(1024→512, t1, s2)。
6. up3: Res(512→256)・Res(256)×2・アップ無し・**外側のショートカット無し**（`avg_shortcut=None`）。
7. head: RMS_norm → SiLU → CausalConv3d(256→12) → unpatchify → `clamp(-1,1)`（`vae2_2.py:1045`）。

モジュールの数（観測 `vae_meta.py`）: デコーダ CausalConv3d 34（うち 1×1×1 の shortcut 2）・Conv2d 5・RMS_norm 30・
Residual 14・DupUp3D 3・attention 1。エンコーダ CausalConv3d 26・Conv2d 5・AvgDown3D 4・ZeroPad2d 3。

AvgDown3D / DupUp3D の中身（`vae2_2.py:316-412`）:

- AvgDown3D: 時間軸の先頭へ `(factor_t - T % factor_t) % factor_t` 枚ゼロを詰める → rank 8 の view
  `[B,C,T/ft,ft,H/fs,fs,W/fs,fs]` → permute `(0,1,3,5,7,2,4,6)` → `[B, C·ft·fs², T', H', W']` →
  `[B, Cout, group, T', H', W']` で `mean(dim=2)`。group = Cin·ft·fs²/Cout = 4（down0〜2）。
- DupUp3D: `repeat_interleave(repeats, dim=1)`（repeats = Cout·ft·fs²/Cin: up0 / up1 は 8・up2 は 2）→ rank 8 の
  view `[B,Cout,ft,fs,fs,T,H,W]` → permute `(0,1,5,2,6,3,7,4)` → `[B,Cout,T·ft,H·fs,W·fs]`。最初の chunk では
  `x[:, :, ft-1:]`（時間 2 倍の先頭 1 枚を捨てる）。

### 3.3 causal conv の cache とフレームの刻み（一次）

- CausalConv3d・`CACHE_T = 2`・cache の持ち回し（入力末尾 2 フレーム、足りなければ前回 cache の末尾 1 枚を前に足す）
  は 2.1 と同じコード形（`vae2_2.py:14-42`・`193-235`）。
- **decode**: 潜在 1 フレームずつ（`vae2_2.py:819-836`）。最初の chunk は `first_chunk=True` で、upsample3d の
  `time_conv` を `'Rep'` の番兵で飛ばし、DupUp3D は先頭を切って 1 フレームを出す。2 chunk 目以降は 4 フレーム。
  合計 1 + 4·(T_lat − 1)（121 フレーム = 1 + 4×30）。diffusers も同じ（`autoencoder_kl_wan.py:1187-1214`）。
- **encode**: patchify の後、フレームを 1, 4, 4, … で刻む（`iter_ = 1 + (t-1)//4` — `vae2_2.py:785-802`、diffusers
  `autoencoder_kl_wan.py:1133-1143`）。downsample3d は最初の chunk で `time_conv` を飛ばして入力を cache に置き、
  以降は cache の末尾 1 枚を前に cat して stride 2 で畳む（`vae2_2.py:157-168`）。
- 公式は VAE を float32 で回す（`Wan2_2_VAE` の `dtype=torch.float` — `vae2_2.py:897`）。Diffusers 版の README の
  例も `AutoencoderKLWan.from_pretrained(..., torch_dtype=torch.float32)`。

### 3.4 形・cache・MAC（観測: meta device）

`uv run --group wan --inexact python vae_meta.py <H> <W> [shapes]`（diffusers 0.39.0 の `AutoencoderKLWan` を
HF の `vae/config.json` で meta device に組み、hook で conv の MAC と最大テンソルを数えた。mid の attention の
MAC は含まない）。

| 条件                              | decode chunk 0（1 フレーム） | decode chunk 1 以降（4 フレーム） | 最大テンソル（chunk 1）                                            | decode の cache 合計（32 本）      |
| --------------------------------- | ---------------------------- | --------------------------------- | ------------------------------------------------------------------ | ---------------------------------- |
| 1280×704（潜在 44×80）            | 10.26 T MAC                  | 33.15 T MAC                       | up2 の nearest 出力 `[4,512,352,640]` 461.4 M 要素（f32 1.72 GiB） | 1,607,936,000 要素（f32 5.99 GiB） |
| 832×480（潜在 30×52）             | 4.55 T MAC                   | 14.69 T MAC                       | 同 `[4,512,240,416]` 204.5 M 要素                                  | 712,608,000 要素（f32 2.65 GiB）   |
| 512×512（潜在 32×32・タイル想定） | 2.99 T MAC                   | 9.64 T MAC                        | 同 `[4,512,256,256]` 134.2 M 要素                                  | 467,763,200 要素（f32 1.74 GiB）   |

- decode 全体（観測: 計算 = chunk 0 + (T_lat−1)×chunk 1）: 1280×704×121 で 1,004.8 T MAC（約 2.0 PFLOP）、
  1280×704×81 で 673.3 T MAC、832×480×81 で 298.4 T MAC。2.1 の 832×480×81 は約 274 TFLOP = 約 137 T MAC
  （2.1 調査 §3.9）で、同じ出力画素なら 2.2 は約 2.2 倍。
- encode（I2V は画像 1 枚 = chunk 0 だけ）: 1280×704 で 2.09 T MAC・最大テンソル 36.2 M 要素（down0 の ZeroPad2d
  出力）。4 フレームの chunk は 6.29 T MAC。
- cache の内訳（潜在 32×32 のタイル・chunk 1 の後）: conv_in 入力 `[48,2,32,32]` 1 本、1024 ch・32×32 が 11 本、
  1024 ch・64×64 が 7 本、1024 ch・128×128 が 1 本、512 ch・128×128 が 5 本、512 ch・256×256 が 1 本、
  256 ch・256×256 が 6 本、1×1×1 shortcut の 2 本は None（`vae_meta.py ... shapes` の出力）。

### 3.5 使う演算（一次から列挙）

| 演算                                                                                | デコーダ                                                      | エンコーダ                                        | 2.1 との差                       |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------- | -------------------------------- |
| conv3d（k3・k1・時間 (3,1,1)）                                                      | 34 本（時間 (3,1,1) は up0 / up1 の `time_conv`）             | 26 本（`time_conv` は stride (2,1,1)・padding 0） | 同種（数と ch が違う）           |
| conv2d 3×3                                                                          | アップサンプル後 3 本 + attention の 1×1（to_qkv / proj）2 本 | stride 2 の 3 本 + 1×1 の 2 本                    | アップ側がチャネルを半分にしない |
| nearest-exact ×2（空間・`Upsample.float()`）                                        | 3 回                                                          | —                                                 | 同じ                             |
| RMS_norm（L2 normalize × √C × γ）                                                   | 30                                                            | 22                                                | 同じ                             |
| SiLU                                                                                | あり                                                          | あり                                              | 同じ                             |
| SDPA 単一 head                                                                      | mid 1（D 1024）                                               | mid 1（D 640）                                    | D が 384 → 1024 / 640            |
| cat / slice（cache）                                                                | あり                                                          | あり                                              | 同じ                             |
| 時間インターリーブ（rank 6 の reshape + stack）                                     | up0 / up1                                                     | —                                                 | 同じ                             |
| **patchify / unpatchify（rank 7 の rearrange）**                                    | 出口（12 → 3 ch）                                             | 入口（3 → 12 ch）                                 | 新規                             |
| **DupUp3D（channel の repeat_interleave + rank 8 の view / permute + 先頭 slice）** | 3                                                             | —                                                 | 新規                             |
| **AvgDown3D（時間の前ゼロ詰め + rank 8 の view / permute + mean）**                 | —                                                             | 4（1 本は恒等）                                   | 新規                             |
| ZeroPad2d((0,1,0,1))（非対称ゼロ詰め）                                              | —                                                             | 3                                                 | 2.1 のエンコーダにもある         |
| clamp(-1,1)                                                                         | 出口                                                          | —                                                 | 同じ                             |

- diffusers の patchify の並びは公式と同じ `(c, r, q)`（`autoencoder_kl_wan.py:931-935` の permute
  `(0,1,6,4,2,3,5)`）。ただし diffusers は当初 `(c, q, r)` で、2025-08-01 の commit `58d2b10a`
  「[wan2.2] fix vae patches」で直った（GitHub API で diff を取得 — 一次）。0.39.0 は直った後の形（観測）。
- diffusers の RMS_norm は fp16 / bf16 入力のとき normalize を f32 で行う（`autoencoder_kl_wan.py:200-206`）。

## 4. サンプリング

### 4.1 既定値（一次）

| 項目                     | 値                                                                                                                                                             | 根拠                                                                 |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| スケジューラ             | UniPC（`FlowUniPCMultistepScheduler`）。選択肢に `dpm++`（`FlowDPMSolverMultistepScheduler`）                                                                  | `textimage2video.py:335-354`、`generate.py:202-207`                  |
| steps                    | 50（config）。`i2v()` の関数既定は 40 だが CLI は config の 50 を渡す                                                                                          | `wan_ti2v_5B.py:34`、`textimage2video.py:420`、`generate.py:84-85`   |
| shift                    | 5.0。i2v の docstring に「480p なら 3.0 を推奨」                                                                                                               | `wan_ti2v_5B.py:33`、`textimage2video.py:439`                        |
| guide                    | 5.0                                                                                                                                                            | `wan_ti2v_5B.py:35`                                                  |
| フレーム                 | 121（config）。docstring は「4n+1」。公式コードに 4n+1 の assert は無い                                                                                        | `wan_ti2v_5B.py:36`、`textimage2video.py:258`                        |
| fps                      | 24（保存も `cfg.sample_fps`）→ 121 フレームで約 5.04 秒                                                                                                        | `wan_ti2v_5B.py:32`、`generate.py:551-554`                           |
| 解像度（T2V）            | `1280*704` / `704*1280` の 2 つだけ（`--size` は width*height）                                                                                                | `configs/__init__.py:46`（`SUPPORTED_SIZES`）、`generate.py:100-102` |
| 解像度（I2V）            | 画像の縦横比で `best_output_size(iw, ih, 32, 32, max_area=704·1280)`                                                                                           | `textimage2video.py:462-474`、`utils.py:202-225`                     |
| negative prompt          | 空なら `sample_neg_prompt`（2.1 と同じ中国語の長文）                                                                                                           | `shared_config.py:19`、`textimage2video.py:293-294`                  |
| Diffusers 版 scheduler   | `UniPCMultistepScheduler`・`flow_shift 5.0`・`use_flow_sigmas true`・`solver_order 2`・`bh2`・`predict_x0`・`final_sigmas_type zero`・`lower_order_final true` | HF `scheduler/scheduler_config.json`                                 |
| Diffusers 版 README の例 | 704×1280・121 フレーム・50 steps・guide 5.0・negative は公式の長文を明示・`export_to_video(fps=24)`                                                            | HF Diffusers 版 `README.md:153-185`                                  |

- スケジューラのコードは Wan2.1 と diff 0（`fm_solvers_unipc.py` / `fm_solvers.py` — 観測）。σ 列の作り方の公式 /
  diffusers の違い（公式 `linspace(0.999,0,51)[:-1]`・diffusers `linspace(1,0.001,51)[:-1]`）と shift 5 の末尾 2
  ステップの timestep（公式 172 / 92・diffusers 175 / 96）は 2.1 調査 §3.5 の値がそのまま当たる。

### 4.2 形と seq_len（一次）

- ノイズ: `randn [48, (F-1)//4+1, H//16, W//16]` f32（`textimage2video.py:285-287`・`311-320`・`488-494`）。
  1280×704×121 で `[48,31,44,80]`。
- seq_len: T2V は `ceil(h·w/(2·2) · T / sp) · sp`、I2V は `T · (H/16) · (W/16) / 4` を sp の倍数へ切り上げ
  （`textimage2video.py:289-291`・`479-483`）。sp = 1（単 GPU）ならトークン数そのもので、パディングは出ない。
- diffusers は高さ・幅を `vae_scale_factor_spatial × patch = 32` の倍数へ切り下げて警告し（`pipeline_wan.py:505-515`）、
  フレーム数は `num_frames % 4 != 1` なら 4k+1 へ丸める（`pipeline_wan.py:493-497`）。ComfyUI の
  `Wan22ImageToVideoLatent` も幅・高さは step 32・既定 1280×704・長さ既定 49（`comfy_extras/nodes_wan.py` の
  1426-1428 行・master `f1072eb0`、二次）。

### 4.3 I2V の画像前処理（一次）

- 公式: `best_output_size` で 32 の倍数の (ow, oh) を選ぶ（幅優先と高さ優先の 2 案のうち縦横比のずれが小さい方）→
  `scale = max(ow/iw, oh/ih)` で LANCZOS リサイズ（覆う側）→ 中央クロップ → `to_tensor` を `[-1,1]` へ
  （`textimage2video.py:462-477`）。例（観測: 計算）: 1920×1080 の画像は (1248, 704)。
- diffusers: `video_processor.preprocess(image, height, width)`（`pipeline_wan_i2v.py:710`）。`resize_mode` の既定は
  `"default"`（`image_processor.py:612`）で、指定の高さ・幅へ LANCZOS で直接リサイズする（クロップしない）。
  高さ・幅は呼び出し側が決める（docstring の例は `max_area` と縦横比から `mod_value` の倍数を計算 —
  `pipeline_wan_i2v.py:70-74`）。**公式とは前処理が違う**。

### 4.4 トークン数と計算量（観測: 計算 `node calc.mjs`）

linear は Q/K/V/O・FFN（トークン数比例）と cross の K/V（文脈 512 行）、attention は QK と PV の MAC。norm・
活性化・time MLP は除く（`perTokTime` だけ別に示す）。

| 条件                     | 潜在           | S      | linear   | self-attn | cross  | 1 forward                 | トークンごとの time MLP | `e0` f32 | FFN 中間 f32 |
| ------------------------ | -------------- | ------ | -------- | --------- | ------ | ------------------------- | ----------------------- | -------- | ------------ |
| 2.2 1280×704×121（既定） | [48,31,44,80]  | 27,280 | 118.72 T | 137.17 T  | 2.57 T | 258.46 T MAC（517 TFLOP） | 1.824 T                 | 1.87 GiB | 1.457 GiB    |
| 2.2 1280×704×81          | [48,21,44,80]  | 18,480 | 80.51 T  | 62.95 T   | 1.74 T | 145.20 T                  | 1.235 T                 | 1.27 GiB | 0.987 GiB    |
| 2.2 832×480×121          | [48,31,30,52]  | 12,090 | 52.77 T  | 26.94 T   | 1.14 T | 80.86 T                   | 0.808 T                 | 0.83 GiB | 0.646 GiB    |
| 2.2 832×480×81           | [48,21,30,52]  | 8,190  | 35.84 T  | 12.36 T   | 0.77 T | 48.98 T                   | 0.547 T                 | 0.56 GiB | 0.437 GiB    |
| 2.2 640×352×81           | [48,21,22,40]  | 4,620  | 20.35 T  | 3.93 T    | 0.44 T | 24.72 T                   | 0.309 T                 | 0.32 GiB | 0.247 GiB    |
| 2.1 832×480×81（参考）   | [16,21,60,104] | 32,760 | 41.04 T  | 98.91 T   | 1.55 T | 141.49 T                  | （[B,6,C] なので不要）  | —        | 1.093 GiB    |

- 2.1 の行は 2.1 調査 §3.9 の 141.5 T MAC と一致する（同じ式の再現）。
- self-attn の S（f32 格納）を B570 の束縛上限 2,147,483,644 B で行ブロックに割ると（観測: 計算 — ADR 0118
  決定 6 と同じ式）、S = 27,280（H = 24）は 1 行 2,618,880 B・820 行 × 34 枚、18,480 は 16 枚、8,190 は 3 枚、
  2.1 の 32,760（H = 12）は 24 枚。cross の S は 27,280 で 1,340,866,560 B（1 枚に収まる）。

## 5. テキストエンコーダ（umT5-XXL）

- 公式: `t5_checkpoint = models_t5_umt5-xxl-enc-bf16.pth`・`t5_tokenizer = google/umt5-xxl`・bf16・text_len 512
  （`wan_ti2v_5B.py:12-13`、`shared_config.py:9-11`）。
- **公式形式の .pth は Wan2.1 と同一**: `Wan-AI/Wan2.2-TI2V-5B` と `Wan-AI/Wan2.1-T2V-1.3B`
  （@ `37ec512624d61f7aa208f7ea8140a131f93afc9a`）の `models_t5_umt5-xxl-enc-bf16.pth` はどちらも
  11,361,920,418 B・LFS sha256 `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d`（HF API — 一次）。
- **Diffusers 版は dtype が違う**:

|                                             | Wan2.1-T2V-1.3B-Diffusers       | Wan2.2-TI2V-5B-Diffusers                                |
| ------------------------------------------- | ------------------------------- | ------------------------------------------------------- |
| `text_encoder/config.json` の `torch_dtype` | float32                         | bfloat16（他のキーは同一 — `diff` で観測）              |
| シャード                                    | 5 本・計 22,723,671,744 B       | 3 本（4,935,812,536 + 4,983,103,192 + 1,442,935,480 B） |
| テンソル                                    | 242・全 F32・5,680,910,336 要素 | 242・全 BF16・5,680,910,336 要素（名前と形は全一致）    |

- 値の突き合わせ（観測 `node cmp.mjs`・HTTP Range で先頭を読んだ）: `encoder.final_layer_norm.weight` 4,096 値・
  `encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight` 2,048 値・`shared.weight` 先頭 8,192 値・
  `encoder.block.23.layer.1.DenseReluDense.wo.weight` 先頭 65,536 値の**計 79,872 値すべてで**、2.2 の bf16 が
  2.1 の f32 の最近接偶数丸め（RNE）とビット一致した。切り捨て丸めとは約半数しか合わない。2.1 の f32 は下位 16 ビット
  が 0 の値が 1 つも無く、bf16 を広げたものではない。
- トークナイザ: Diffusers 版の `tokenizer/` は 2.1 と 2.2 で sha 一致（`tokenizer.json` 16,837,459 B
  `20a46ac2…`・`spiece.model` 4,548,313 B `e3909a67…`）。公式形式の `google/umt5-xxl/tokenizer.json` は
  16,837,417 B `6e197b4d…` で別ファイル（2.1 公式形式とは一致）。`google/umt5-xxl` 本家
  （@ `66cb9e7e85526fe440a945569e42c72fb6cbc0ad`・apache-2.0）の `tokenizer.json` は 16,853,013 B `af904105…` で
  さらに別。`spiece.model` は 4 か所すべて同じ sha（HF API — 一次）。
- ソース: Wan2.1 と Wan2.2 の `wan/modules/t5.py`・`wan/modules/tokenizers.py` は diff 0（観測）。
- `google/umt5-xxl` 本家は encoder-decoder の `pytorch_model-0000N-of-00006.bin`（pickle・計約 51.9 GB）だけで、
  safetensors は無い（HF API — 一次）。

推測:

- （推測）2.2 の Diffusers 版 bf16 は 2.1 の Diffusers 版 f32 を RNE で bf16 に落としたもの。抜き取りは 4 テンソルの
  先頭だけなので、全要素の一致は確かめていない。Karume の Wan2.1 のテキスト埋め込み（2.1 の f32 を CPU bf16 で回す
  — ADR 0118 決定 4）は、torch の bf16 キャストが RNE なら 2.2 の bf16 重みで回したものと同じ重みを使う。
- （推測）2.1 の f32 の出どころ（`google/umt5-xxl` の f32 の encoder 部分か）は確かめていない（§10）。

## 6. ライセンス・pin 候補・ファイルサイズ

- ライセンス: Apache 2.0。HF cardData `license: apache-2.0`（両リポ）、gated なし、GitHub `LICENSE.txt` が
  Apache License 2.0、README「The models in this repository are licensed under the Apache 2.0 License」
  （`README.md:496-497`）。
- pin 候補（HF API の `sha` — 一次）:

| リポ                              | commit                                     | lastModified         |
| --------------------------------- | ------------------------------------------ | -------------------- |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `b8fff7315c768468a5333511427288870b2e9635` | 2025-08-09T03:35:40Z |
| `Wan-AI/Wan2.2-TI2V-5B`           | `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | 2025-08-07T10:22:24Z |
| GitHub `Wan-Video/Wan2.2`         | `1ea34ff48f87168174e12956e200b1d908b1c5ff` | 2026-09-21           |

- ファイル（Diffusers 版・バイト・LFS sha256 の先頭 8 桁）:
  - `transformer/` 5 本: 4,978,254,344（`511bec83`）・4,846,784,976（`7c427249`）・4,972,658,392（`e9c3d0c7`）・
    4,846,785,080（`a3311217`）・354,751,840（`78b65568`）。index 73,297。
  - `vae/diffusion_pytorch_model.safetensors` 2,818,777,808（`62cd18f1`）。
  - `text_encoder/` 3 本（§5）。`tokenizer/` 4 本。`scheduler/scheduler_config.json` 820・`model_index.json` 499・
    `README.md` 17,633。
  - safetensors と pth の合計 34,179,863,648 B。
- ファイル（公式形式）: `diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors` 9,825,014,472（`720b06c4`）・
  9,995,661,736（`09ec5ef7`）・178,558,176（`6306f789`）、`Wan2.2_VAE.pth` 2,818,839,170（`20eb7896`）、
  `models_t5_umt5-xxl-enc-bf16.pth` 11,361,920,418（`7cace0da`）、`google/umt5-xxl/*`。合計 34,179,993,972 B。

## 7. Diffusers 版の構成と export の上流

- `model_index.json`: `_class_name: WanPipeline`・`expand_timesteps: true`・`boundary_ratio: null`・
  `transformer_2: [null, null]`・部品は `WanTransformer3DModel` / `AutoencoderKLWan` / `UMT5EncoderModel` /
  `T5TokenizerFast` / `UniPCMultistepScheduler`。
- diffusers 側の対応（0.39.0 — 観測）:
  - T2V: `WanPipeline(expand_timesteps=True)`（`pipeline_wan.py:140`）。マスクは全 1（`pipeline_wan.py:574`）で、
    タイムステップは `(mask[0][0][:, ::2, ::2] * t).flatten()`（`pipeline_wan.py:606-610`）。
  - I2V: `WanImageToVideoPipeline` の `expand_timesteps` 分岐（§2.3）。`image_encoder` は optional
    （`pipeline_wan_i2v.py:165`）。
  - VAE: `AutoencoderKLWan(is_residual=True, patch_size=2, ...)`（`autoencoder_kl_wan.py:535-557`・`814-852`・
    `1022-1025`）。Diffusers 版の `vae/config.json` の `clip_output: false` は 0.39.0 の `__init__` に無い引数で、
    `AutoencoderKLWan(**config)` は TypeError になった（観測。`from_pretrained` は未知キーを捨てる（推測）ので
    読み込みは通る見込み）。
- HF Diffusers 版の README は「diffusers の main が要る」と書く（2025-08 時点・`README.md:188-192`）。0.39.0 は
  TI2V の分岐を全て持つ（上の行番号で観測）。

比較:

| 観点           | Diffusers 版                                                                                                 | 公式形式                                                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| 形式           | 全部品 safetensors（VAE も）                                                                                 | DiT は safetensors、VAE と T5 は pickle の .pth（`torch.load` が要る）                                                   |
| DiT の切り出し | Karume の 2.1 recipe と同じ `WanTransformer3DModel`（テンソル入力・SDPA）。重み名も 2.1 Diffusers と同じ規則 | `WanModel` は torch.export に向かない（2.1 調査 §3.10 と同じ理由: リスト入力・`grid_sizes.tolist()`・flash_attn 直呼び） |
| DiT の値       | 公式とビット一致（抜き取り）                                                                                 | —                                                                                                                        |
| VAE            | `AutoencoderKLWan` の `is_residual` 分岐（2.1 と同じクラス）                                                 | `Wan2_2_VAE`（別ファイル `vae2_2.py`）                                                                                   |
| umT5           | bf16 3 本・11.36 GB（2.1 Diffusers の f32 22.7 GB の半分）                                                   | bf16 .pth 11.36 GB                                                                                                       |
| スケジューラ   | `flow_shift 5.0` が config に入る                                                                            | config の `sample_shift`                                                                                                 |

推測:

- （推測）export の上流は Diffusers 版が扱いやすい。Karume の Wan2.1 recipe の DiT / VAE のパッチと参照パイプラインの
  骨格を、クラスの分岐の差（トークンごとの t・`is_residual`・patchify）だけ足して流用できる見込み。umT5 の取得量も
  半分になる。

## 8. 低 VRAM 運用の実例（二次ソースを含む）

一次（開発元）:

- 公式 README: 単 GPU の TI2V は `--offload_model True --convert_model_dtype --t5_cpu`（DiT を bf16 化・T5 は CPU）で
  「24GB VRAM（RTX 4090）で動く」（`README.md:229-236`）。80GB 以上ならこれらを外して速くできる。
- 公式の効率表（HF `assets/comp_effic.png` の画像を読んだ — 観測）: RTX 4090・720P・1 GPU で TI2V-5B T2V
  534.7 s / ピーク 22.9 GB、I2V 524.8 s / 22.8 GB。4 GPU で 231.3 s / 227.3 s、8 GPU で 157.2 s / 160.1 s。条件は
  単 GPU が上記 3 フラグ・prompt extend 無し・ウォームアップ後の平均（`README.md:434-441`）。README 本文は
  「5 秒の 720P を単一の消費者向け GPU で 9 分未満」（`README.md:467`）。

二次:

- ComfyUI 公式ドキュメント（docs.comfy.org/tutorials/video/wan/wan2_2 を WebFetch — 準一次。ComfyUI は開発元では
  ない）: 「The Wan2.2 5B version should fit well on 8GB vram with the ComfyUI native offloading.」。使うファイルは
  `wan2.2_ti2v_5B_fp16.safetensors`・`umt5_xxl_fp8_e4m3fn_scaled.safetensors`・`wan2.2_vae.safetensors`。
- Comfy-Org の再梱包（HF `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` @ `ee6f4a40` の API）: DiT fp16 9,999,658,848 B・
  umT5 fp16 11,366,399,385 B・umT5 fp8 e4m3fn scaled 6,735,906,897 B・VAE 1,409,400,960 B（= 704.7 M × 2 B で
  f16 と推測）。
- GGUF: `QuantStack/Wan2.2-TI2V-5B-GGUF`（@ `57437632`・2025-07-31）で Q8_0 5,400,179,040 B・Q6_K 4,211,683,680 B・
  Q4_K_M 3,433,116,000 B・Q3_K_M 2,547,790,176 B・Q2_K 1,853,862,240 B（HF API）。HF の検索ではダウンロード上位に
  `unsloth/Wan2.2-TI2V-5B-GGUF`・`hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF`、少ステップ蒸留の
  `quanhaol/Wan2.2-TI2V-5B-Turbo`・`yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers` がある（中身は未確認）。
- HF フォーラム（discuss.huggingface.co/t/.../170034 を WebFetch・著者不明）: 8GB で GGUF を Unet Loader (GGUF) で
  読み、672×384・33 フレーム・24 fps・batch 1・12〜16 steps・CFG 約 2.0・VRAM モード auto / lowvram から始める、
  という手引き。672×384・33 フレームは 2,268 トークン（観測: 計算）。
- 検索結果の要約（WebSearch・二次、出典の個別検証なし）: Kijai の fp8 版で 5B が 5.28GB になる、という記述。

## 9. 推測と含意（Karume への移植）

本節は全て推測（設計判断の材料。裁定ではない）。

- **DiT**: Wan2.1 の S 形グラフと recipe パッチ（patchify の Linear 化・RoPE の実数化・SDPA の保存）は寸法を
  変えるだけで当たる見込み。新しいのはトークンごとの変調だけ。T2V は `[1,6,C]` の broadcast で済み、I2V は t の値が
  2 種なので「time MLP を 2 行だけ回す + 先頭フレームのトークン範囲で振り分ける」形にすれば、`e0` の 1.87 GiB と
  1.8 T MAC を避けられる。グラフの入出力をどう切るか（2 行の入力 + トークン数 P = (H/32)·(W/32) で slice / cat
  するか、ホストで S 行に展開するか）は ADR の分岐点。
- **DiT の容量**: f16 の重みだけで 9.31 GiB あり、B570（VRAM 9.93 GiB —
  [2026-10-02 の測定](2026-10-02-b570-vram-budget.md)）に f16 席では載らない見込み。Linear だけ i8 なら約 4.66 GiB。
  w8a8 席（ADR 0120）が実用の前提になる。
- **計算量**: 同じ出力画素なら 2.1 1.3B より DiT は軽い（832×480×81 で 0.35 倍）。既定の 1280×704×121 では
  1 forward が 2.1 の 832×480×81 の 1.83 倍・self-attn の行ブロックは 34 枚。
- **VAE**: 新しい演算は patchify / unpatchify・DupUp3D・AvgDown3D（エンコーダ）で、どれもデータ移動 + 平均 / 複製。
  rank 7〜8 の view / permute は rank 4 以下への書き直しが要る。DupUp3D は「チャネルの複製 + 3D の depth-to-space」、
  AvgDown3D は「3D の space-to-depth + グループ平均」と読める。
- **VAE の容量**: 1280×704 の全画面 decode は cache だけで f32 5.99 GiB、最大中間 1.72 GiB で、タイルが前提。
  潜在 32×32 のタイル（出力 512 px）でも cache が f32 1.74 GiB で、ADR 0118 の「入力 + 出力 + first の出力」の 3 組
  だと約 5.2 GiB になる。タイルを潜在 16×16 に縮める（cache 約 0.44 GiB）か、cache の f16 格納などの手が要る。
  デコーダの mid の attention の D は 1024 で、Karume の融合 attention がこの D を受けるかは未確認（Anima は D 384）。
- **I2V のエンコーダ**: 画像 1 枚 = encode の chunk 0 だけなので、エンコーダは `time_conv` 無し・T = 1 のグラフ 1 本で
  済む見込み（downsample3d は最初の chunk で `time_conv` を飛ばす）。
- **umT5**: Wan2.1 の umT5 の資産（埋め込みの事前計算・GPU 化）は重みが同じなので流用できる見込み。モデルカードの
  出どころの書き方（Wan 由来か google 本家か）は別の調査。

## 10. 取得に失敗したもの・未確認

- 取得の失敗は無かった（GitHub raw / API・HF API / HTTP Range・WebFetch 2 件はすべて成功）。
- 未確認:
  - Diffusers 版 umT5 の bf16 と 2.1 の f32 の全要素の一致（抜き取り 79,872 値だけ）。2.1 Diffusers の f32 と
    `google/umt5-xxl` の f32（pickle の .bin）の一致。
  - `WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers")` が通るか（実行していない）。
  - 公式 VAE の `.pth` と Diffusers 版 VAE の値の一致（pth は pickle で Range 読みしていない）。
  - 少ステップ蒸留版（Turbo 系）の構成・ライセンス。
  - 二次ソースの低 VRAM の設定値の実測。

## 11. 実行したコマンド（再現用）

作業場所は `/tmp/claude-1000/recon-2026-10-04/wan22/`（リポ外・消して安全）。

- `git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git` → HEAD `1ea34ff4…`。
- `curl "https://huggingface.co/api/models/<repo>?blobs=true&files_metadata=true"` で sha・ファイル・LFS sha256。
- `node hdr.mjs <repo> <rev> <path> <out>`: `Range: bytes=0-7` で長さを読み、続けてヘッダ JSON を読む。
- `node ana.mjs hdr/*.json`: テンソル数・要素数・dtype・ブロック内訳・2D 重みの数。
- `node cmp.mjs` / `node cmp2.mjs` / `node low16.mjs`: テンソルの先頭を Range で読んでビット比較。
- `uv run --group wan --inexact python vae_meta.py 704 1280` / `480 832` / `512 512 shapes`（tools で実行・meta device）。
- `node calc.mjs`: トークン数・MAC・中間の大きさ。
- `diff` で Wan2.1（main を raw 取得）と Wan2.2 の `model.py` / `t5.py` / `tokenizers.py` / `fm_solvers*.py` を比較。

## 独立検証（2026-10-04）

別のレッグが、設計が依存する主張 13 個を一次ソースと再計測で確かめた。成り立つ（holds）が 13 個。

### 本文が落としていた、設計に効く事実

- 公式は --convert_model_dtype が無くても DiT の forward を常に bf16 autocast で回す（textimage2video.py:330・522、param_dtype=bf16）。重みは F32 でも上流の数値の基準は bf16 なので、参照パイプラインの dtype と sha 参照値の決め方に効く
- DiT の小さいテンソル（518 本）を全要素読むと |w| の最大は 7.71（blocks.29.attn1.norm_k）。文書の「抜き取りで最大 0.25」は Linear の先頭だけの値。f16 の範囲には収まるが、norm の γ が大きい点は f16 席の活性値の溢れを見積もるときの材料
- Diffusers 2.1 の umT5 f32 は下位 16 ビットが 0 の値が無く、公式 bf16 .pth を広げたものではない。公式 .pth の bf16 が f32 を RNE したものかは確かめていない。どちらを『正』の重みとするかはモデルカードの出所と umT5 差し替えの設計に効く
- 2.4 の推測（公式の step 後置換と diffusers の入力だけ置換は先頭フレーム以外で同値）は、コードを読む限り成り立つ見込み。UniPC の sample は要素ごとに扱われ、CFG も同じ入力に掛かる。ただし実測はしていない（推測）
- Wan2.2 の model.py:460-461 の t.expand(t.size(0), seq_len) は 1 次元 t で B>1 だと形が合わない（パイプラインは常に [1,S] を渡すので実害なし）。I2V 経路をバッチで組むなら diffusers 側の形を基準にすべき
