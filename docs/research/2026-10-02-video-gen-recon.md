# 動画生成モデル recon — Wan2.1 T2V 1.3B と MiniMax H3（2026-10-02 時点）

> NOTE: 2026-10-02 時点の調査記録（時点スナップショット）。裁定は含まない — 正本は ADR と
> backlog。手法: 並列調査 7 レッグ（掃引 4 本: Wan2.1 1.3B の中身 / MiniMax H3 / karume の受け入れ
> 状況 / 先行事例と相場、深掘り 3 本: Wan の op と形状の棚卸し / H3 の敵対的裏取り / karume に無い
> op の実装コスト）の戻り値を統合した。GPU 実測は無い。重みは取得しておらず、読んだのは
> safetensors ヘッダ（HTTP Range）・config・ソース・ドキュメントだけ。

記法: 根拠は URL・`file:line`・「観測」（こちらでヘッダ集計やスクリプト計算をした値）のどれかで
示す。「一次」は開発元のソース・配布物・公式ドキュメント、「二次」はそれ以外。推測には（推測）を
付ける。Wan 公式ソースの `file:line` は Wan-Video/Wan2.1 の main（取得時点）、diffusers の
`file:line` は huggingface/diffusers の main（取得時点）を指す。karume の `file:line` の基準点は
コミット 3010359c + 調査時点の作業ツリー。`docs/limitations.md` は後のコミット 5dec78f0 で L235 付近に
1 行挿入されたので、それより後ろの行を指す引用は現行の HEAD では 1 行ずれる。

先行調査: [2026-09-10 の事前調査](2026-09-10-codex-mtp-optimization.md#動画生成の事前調査-wan-と-minimax-h3)
（`docs/research/2026-09-10-codex-mtp-optimization.md:409-453`）と `docs/backlog.md:322-325`
（「ブラウザ動画生成の基盤」）に方針の骨子が既にある。本記録はそれをコードと一次ソースで裏付け、
数値を詰めたもの。2026-09-10 の事前調査が置いた検収順序は、① 小さな固定 text 条件で DiT の CPU/GPU
照合 → ② 実 token 長の attention / FFN の容量と時間 → ③ causal VAE の短い時間 chunk → ④ 公式
scheduler と各段を結ぶ経路（同 `:435-436`）。

## 1. 問い

利用者の要望は 2 段。

- まず Wan2.1 T2V 1.3B（Alibaba の text-to-video モデルの最小版）を karume で動かしたい。
- 最終目標は MiniMax H3（MiniMax の動画 + 音声生成モデル）。

調べたのは次の 4 点。Wan2.1 1.3B の構成と数値、karume が今それをどこまで受けられるか、足りない
部品を足すコスト、H3 が実在して手元で動かせる対象か。

## 2. 結論の要約

- Wan2.1 1.3B の DiT（拡散 Transformer 本体）は、karume の現行語彙でほぼ足りる。新しい op は要らず、
  recipe のパッチ（complex 形 RoPE の実数化・unpatchify の rank 下げ）で済む見込み（§4.1）。
- 空白は 4 つ。動画 VAE の conv3d と因果キャッシュ（feat_cache）の持ち越し（§4.3・§5）、umT5-XXL
  （5.68B パラメータのテキストエンコーダ）の載せ方（§6）、32,760 トークンの全結合 attention の
  計算時間と TDR（GPU のタイムアウト検出）（§4.5・§7）。
- 容量は分割で収まる見込みだが、計算量は 1 forward 約 283 TFLOP・既定 50 ステップ × CFG 2 回で
  約 28.3 PFLOP と重い（§4.5）。
- MiniMax H3 は実在し、open-weight（HF で gated なし）。ライセンスの条文を文理で読むと、日本は許諾
  地域に入る（法的助言ではない・§8.3）。
- H3 は 33B dense の DiT + Qwen3-VL-32B のテキストエンコーダ + 2.6B の映像 VAE で、1 タスクの取得量は
  約 144 GB。ネイティブでも offload が前提で、RTX 5090 で 864×480・124 フレーム・20 steps を 112.2 s
  で回した実測例がある（§8.4）。ブラウザ WebGPU で動かした事例は無い（本調査で確認した範囲。先行事例の
  検索は Wan2.1 が対象）。（推測）ホスト RAM への offload に相当する手段がブラウザに無い点が障壁になる。

## 3. Wan2.1 T2V 1.3B の構成

### 3.1 用語（本節の初出）

- **DiT**: 潜在（VAE で圧縮した動画）をパッチに切ってトークン列にし、Transformer でノイズを除く
  拡散モデル本体。
- **AdaLN**（adaptive LayerNorm）: 時刻 t から作ったベクトルで LayerNorm 後の値を
  `x*(1+scale)+shift` と変調し、残差に gate を掛ける仕組み。
- **flow matching**: ノイズとデータを直線で結ぶ定式化。モデルは速度（ノイズ − データ方向）を予測し、
  サンプラは σ を 1 → 0 へ進める。**shift** は σ 列を高ノイズ側へ寄せるパラメータ。
- **UniPC**: 少ないステップ数向けの多段 ODE ソルバ（ここでは 2 次・bh2 変種）。
- **CFG**（classifier-free guidance）: 条件付きと条件なし（negative prompt）の 2 回 forward を取り、
  `uncond + g·(cond − uncond)` で条件を強める手法。1 ステップの forward が 2 倍になる。
- **causal 3D VAE**: 時間方向は過去フレームだけを見る（因果的な）3D 畳み込みで、動画を潜在へ圧縮・
  復元する VAE。チャンクごとに前チャンクの末尾特徴（feat_cache）を持ち越す。
- **qk-norm**: attention の q と k に RMSNorm を掛けて logits の発散を抑える手法。
- **3D RoPE**: 回転位置埋め込みを時間・高さ・幅の 3 軸に分けて掛けるもの。

### 3.2 DiT の寸法とブロック

| 項目                | 値                                                                  | 根拠                                                  |
| ------------------- | ------------------------------------------------------------------- | ----------------------------------------------------- |
| クラス              | `WanModel`（公式）/ `WanTransformer3DModel`（diffusers）            | `wan/modules/model.py` / `transformer_wan.py`         |
| 層数・hidden・heads | 30 層・dim 1536・12 heads × head_dim 128                            | `wan/configs/wan_t2v_1_3B.py:17-29`、HF `config.json` |
| FFN                 | 1536 → 8960 → 1536、GELU(tanh)                                      | 同上、`model.py:238-317`                              |
| patch               | (1,2,2)（時間 1 × 空間 2×2）、パッチ埋め込みは Conv3d               | `wan_t2v_1_3B.py`、`model.py:456-457`                 |
| attention           | 窓なしの全結合（window (-1,-1)）、マスクなし                        | `wan_t2v_1_3B.py`、`model.py:149-154`                 |
| 入出力ch・text      | in/out_dim 16・text_dim 4096・text_len 512・freq_dim 256・eps 1e-6  | `wan_t2v_1_3B.py:20-29`、`shared_config.py:11`        |
| 重み                | 825 テンソル・全 F32・1,418,996,800 パラメータ（fp16 換算 2.84 GB） | 観測（safetensors ヘッダ集計）                        |
| 内訳                | 1 ブロック 46,440,704 × 30 + 埋め込み・time・head 25,775,680        | 観測（同上）                                          |

HF 原版の `config.json` は patch_size・qk_norm・cross_attn_norm・text_dim を持たず
（`ignore_for_config`）、`WanModel.__init__` の既定値が使われる（`model.py:377-398`）。

1 ブロックは 3 つの残差枝からなる（`model.py:259-273`・`301-316`）。

1. LayerNorm（affine なし・f32）→ `×(1+scale_msa)+shift_msa` → self-attn → 残差に `gate_msa` を
   掛けて加算。
2. LayerNorm（affine あり）→ T5 文脈への cross-attn → 残差にそのまま加算（ゲートなし）。
3. LayerNorm（affine なし）→ `×(1+c_scale)+c_shift` → FFN → 残差に `c_gate` を掛けて加算。

- q/k/v/o は bias 付き Linear。qk-norm は head 分割の前に 1536 次元全体へ掛ける RMSNorm（affine
  付き・eps 1e-6）で、cross-attn の q（x 側）と k（文脈側）にも掛かる（`model.py:73-89`・`127-128`・
  `141-145`・`174-176`。diffusers の名前は `rms_norm_across_heads`）。
- 最終 head は LayerNorm（affine なし）→ 2 成分の AdaLN → Linear 1536→64（= 16ch × 1×2×2）→
  unpatchify（`model.py:320-347`）。unpatchify は公式が 7 次元の einsum
  `fhwpqrc->cfphqwr`（`model.py:584-607`）、diffusers が 8 次元の permute
  （`transformer_wan.py:723-730`）。
- **重み名のねじれ**: affine 付きの cross-attn 前 norm は、公式が `blocks.N.norm3`、diffusers が
  `blocks.N.norm2`。FFN 前の norm は逆になる（`model.py:259-270`、`transformer_wan.py:434-458`、
  観測: 両版の safetensors 名）。

### 3.3 RoPE・時刻埋め込み・テキスト文脈

- **3D RoPE**: head_dim 128 を t/h/w = 44/42/42 次元（複素の対で 22/21/21）に分ける。theta 10000、
  各軸の表は最大位置 1024。適用は self-attn の q/k だけ（v と cross-attn には掛けない）
  （`model.py:31-39`・`42-70`・`478-485`）。
  - 回転の対は隣り合う 2 要素 `(x[2i], x[2i+1])`（interleave 形）。karume の rope 融合が掴む
    half-split 形（`rotate_half`）ではない（§4.3）。
  - 公式は `torch.polar` の複素表と float64 の `view_as_complex` で回す（`model.py:55`）。diffusers は
    float64 で表を作ってから f32 に落とし、`out[...,0::2]` / `out[...,1::2]` への strided 代入で
    適用する（`transformer_wan.py:104-118`・`368-416`、`embeddings.py:929-940`）。
  - 832×480・81 フレームでの cos / sin 表は各 `[1,32760,1,128]` f32 = 16 MiB（観測: 計算）。形が固定
    なら事前計算できる。
- **時刻埋め込み**: t（0〜1000 の整数）→ sinusoidal 256 次元（cos 先・sin 後、公式は float64）→
  Linear 256→1536 → SiLU → Linear で `e` → SiLU → Linear 1536→9216 で `e0 [1,6,1536]`
  （`model.py:18-28`・`462-464`・`545-550`）。
  - この MLP は全ブロックで共有し、各ブロックは学習パラメータ `modulation [1,6,1536]`（diffusers 名
    `scale_shift_table`）を足して 6 分割する（README「This MLP is shared across all transformer
    blocks, with each block learning a distinct set of biases」）。
  - head は SiLU を通す前の `e` に `head.modulation [1,2,1536]` を足す（`model.py:337-347`）。
- **テキスト文脈**: umT5-XXL の出力 `[1,512,4096]` を有効長で切り、ゼロで 512 まで埋め戻してから
  `text_embedding`（Linear 4096→1536 → GELU(tanh) → Linear）に通す（`t5.py:506-513`、
  `model.py:552-559`）。
  - 埋め戻しは射影の**前**なので、DiT が見るパディング位置は**非ゼロの定数ベクトル**になる。掃引レッグ
    の「ゼロ埋めされた 512 トークン」は射影前の話で、深掘りレッグが補正した。
  - cross-attn にマスクは無い（公式 `context_lens=None`・`model.py:553`、diffusers
    `attention_mask=None`）。パディング込みの 512 トークン全体に attend する挙動は、移植で再現が要る。
  - 文脈 `[1,512,1536]` と各ブロックの cross-attn の K/V はステップを跨いで不変なので、プロンプトごと
    に 1 回計算してキャッシュできる（導出）。

### 3.4 精度の扱い

- 公式は DiT 重みを F32 で配布し、bf16 の autocast で回す（`shared_config.py:14`、
  `text2video.py:204`）。変調の加算・ゲート付き残差・head は f32 の autocast に固定し、`e` が f32 で
  あることを assert する（`model.py:296-314`・`343-346`）。型昇格により残差流 x は f32 になる。
- diffusers は加算を f32 で行ってから `.type_as(hidden_states)` で bf16 へ戻す。
  `_keep_in_fp32_modules` は rope / time_embedder / scale_shift_table / norm1-3
  （`transformer_wan.py:488-502`・`549`）。
- （推測）bf16 ではなく f16 で残差流が値域に収まるかは実測が要る。WebGPU には bf16 も f64 も無い
  （§7.4）。

### 3.5 サンプラと既定値

- サンプラは flow matching の UniPC。公式 `FlowUniPCMultistepScheduler` と diffusers
  `UniPCMultistepScheduler(use_flow_sigmas=True)` は、どちらも solver_order 2・bh2・predict_x0・
  最後の σ は 0（`wan/utils/fm_solvers_unipc.py:79-135`、`scheduling_unipc_multistep.py:428-466`）。
- σ 列は `shift·σ/(1+(shift−1)σ)` で作るが、元の等間隔列が違う。公式は `linspace(0.999, 0, 51)[:-1]`、
  diffusers は `linspace(1, 0.001, 51)[:-1]`。timestep は σ×1000 を int64 に切り捨てて DiT へ渡す。
  - 観測（計算）: shift 3 の末尾 2 ステップは公式 111 / 57・diffusers 113 / 60。shift 5 は公式
    172 / 92・diffusers 175 / 96。
- 初期ノイズは CUDA の `torch.Generator` で作る `randn [16,21,60,104]` f32（`text2video.py:168-195`）。
  （推測）CUDA の RNG はプラットフォームを跨いでビット再現できないので、参照出力との照合では
  ノイズを外から注入する経路が要る。
- negative prompt の既定は `shared_config.py` の中国語の長い文字列（`sample_neg_prompt`）。

既定値は出どころで割れる。

| 出どころ                                   | steps | guide | shift                                          | 根拠                                                       |
| ------------------------------------------ | ----- | ----- | ---------------------------------------------- | ---------------------------------------------------------- |
| 公式 `generate.py` の T2V 既定             | 50    | 5.0   | 5.0                                            | `generate.py:70-85`・`242-244`                             |
| 公式 README（1.3B 向け推奨）               | —     | 6     | 8〜12（例のコマンドは 8）                      | README `:177`・`:180`                                      |
| Diffusers 版 HF の `scheduler_config.json` | —     | —     | 3.0（docstring「5.0 for 720P, 3.0 for 480P」） | HF `scheduler/scheduler_config.json`、`pipeline_wan.py:58` |
| diffusers `WanPipeline.__call__` の既定    | 50    | 5.0   | （scheduler 側）                               | `pipeline_wan.py:388-404`                                  |

- 対応サイズは 480×832 / 832×480 のみ（README: 720P も出るが不安定）。フレーム数は 4n+1 で既定 81、
  16 fps で約 5.06 秒（`wan/configs/__init__.py` の `SUPPORTED_SIZES`、`shared_config.py:18`）。
- 「1.3B の推奨は guide 6.0・shift 3.0」という言い方は、guide が公式 README、shift が Diffusers の
  config 由来で、出どころが混ざっている（深掘りレッグの指摘）。

### 3.6 部品のサイズ

| 部品             | パラメータ                                         | 公式の配布形                                       | Diffusers 版                                           | 根拠                        |
| ---------------- | -------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------ | --------------------------- |
| DiT              | 1,418,996,800（fp16 換算 2.84 GB）                 | safetensors F32・5,676,070,424 B                   | `transformer/` 2 分割（4,998,781,576 + 677,289,072 B） | 観測（ヘッダ・HF tree API） |
| VAE              | 126,892,531（うちデコーダ 73,295,331・f32 293 MB） | `Wan2.1_VAE.pth`（pickle）507,609,880 B            | safetensors F32・507,591,892 B                         | 観測（ヘッダ集計）          |
| umT5-XXL encoder | 5,680,910,336（bf16 で 11.36 GB）                  | `models_t5_umt5-xxl-enc-bf16.pth` 11,361,920,418 B | fp32 safetensors 5 分割                                | HF tree API、構成からの計算 |

- Diffusers 版 text_encoder の合計は、index の `total_size` が 22,723,641,344 B、5 ファイルの実
  サイズの和が 22,723,671,744 B（差はヘッダ分と推測）。
- umT5 の語彙埋め込み `256,384 × 4,096` は fp16 で 2,100,297,728 B（1.96 GiB）、f32 で
  4,200,595,456 B（観測: 計算）。
- 公式形式の VAE と T5 は pickle の `.pth` で、読むには `torch.load` が要る。
- Wan の技術レポートは T5 を「5.3B」と書く（arXiv 2503.20314・WebFetch 要約経由）。上の 5.68B は
  config からの計算とファイルサイズの逆算が一致した値。

### 3.7 umT5-XXL encoder の構成

- vocab 256,384・d_model 4096・24 層・64 heads × d_kv 64・FFN は gated-GELU（`fc1(x) *
  gelu_tanh(gate(x))`、4096→10240→4096、bias なし）・T5LayerNorm（平均を引かない RMSNorm・eps
  1e-6）・最後にも norm（`wan/modules/t5.py:46-141`・`456-469`、HF `text_encoder/config.json`）。
- attention は QK^T にスケーリングを掛けない（T5 の慣例・`t5.py:111`）。
- 相対位置バイアスは層ごとに別（shared_pos=False）・32 buckets・双方向・max_dist 128 で、
  `[1,64,L,L]` を作って logits に足す（`t5.py:221-264`）。パディングはマスクで `finfo.min` を足して
  除き、softmax は f32（`t5.py:105-109`）。
- tokenizer は `google/umt5-xxl`。`padding='max_length'`・max_length 512・EOS あり・pad_id 0
  （`tokenizers.py:54-58`）。
- 512 トークンの計算量は約 2.42 T MAC（観測: 計算）。
- 各残差加算に fp16 で inf が出たら `finfo.max-1000` でクランプする `fp16_clamp` が入っている
  （`t5.py:20-24`・`173-174`）。公式は T5 を bf16 で回す（`shared_config.py:10`）。
- 導出（未実測）: 有効トークンはパディングのキーを見ず、相対位置バイアスは距離だけで決まり、出力は
  有効長で切られる。よって有効長 L_valid だけで回しても有効トークンの出力は丸め誤差の範囲で一致する
  はずで、計算量は L_valid に比例して減る（`t5.py:102-113`・`233-243`・`513`）。negative prompt の
  埋め込みは固定なのでキャッシュできる。

### 3.8 VAE デコーダの構成

- 構成値: base_dim 96・z_dim 16・dim_mult [1,2,4,4]・num_res_blocks 2（デコーダは 1 段に 3 個）・
  temperal_upsample [True,True,False]・stride (4,8,8)（`wan/modules/vae.py:369-421`・`592-616`、HF
  `vae/config.json`）。
- 部品は CausalConv3d（時間方向は前に 2p だけゼロ詰め・空間は両側 1）・RMS_norm・SiLU
  （`vae.py:17-63`）。RMS_norm の中身は `F.normalize`（L2 ノルムを `clamp_min(1e-12)` して割る）
  × √C × gamma + bias で、一般の RMSNorm と eps の入れ方が違う（`torch/nn/functional.py:6143-6177`）。
  diffusers は入力が fp16 / bf16 のとき normalize を f32 で行う（`autoencoder_kl_wan.py:202-210`）。
- 流れ（`vae.py:369-421`）:
  - 前処理 `z*std+mean`（16ch 分の固定値・`vae.py:547-551`・`629-639`）→ 1×1×1 conv（16→16）→
    CausalConv3d(16→384, k3)。
  - mid: Res(384) → フレーム単位の単一 head attention(384) → Res(384)。
  - up0: Res(384)×3 → upsample3d（`time_conv` 384→768・k=(3,1,1) でフレーム 2 倍 → nearest ×2 →
    Conv2d 3×3）。
  - up1: Res(192→384)、Res×2 → upsample3d。
  - up2: Res(192)×3 → upsample2d（nearest ×2 → Conv2d 192→96）。
  - up3: Res(96)×3 → RMS_norm → SiLU → CausalConv3d(96→3) → `clamp(-1,1)`。
- デコードは潜在 1 フレームずつ 21 回回す（`vae.py:544-568`）。各 CausalConv3d の入力末尾 2 フレーム
  （`CACHE_T = 2`・`vae.py:14`）を feat_cache に持ち回し、次のチャンクで前に連結する。
  - 最初のチャンクは `'Rep'` という番兵で `time_conv` を飛ばすので出力 1 フレーム。2 チャンク目以降は
    4 フレームずつ出て、合計 1 + 20×4 = 81（`vae.py:101-137`、diffusers は `first_chunk` フラグ・
    `autoencoder_kl_wan.py:1191-1214`）。
- 公式は VAE を float32 で回す（`vae.py:618-663`）。diffusers も「Set the AutoencoderKLWan dtype to
  torch.float32 for better decoding quality」と推奨する（HF diffusers docs の Wan ページ）。
- T2V ではエンコーダは不要。

### 3.9 832×480・81 フレームの形と量

| 量                           | 値                                                                                      | 根拠                                  |
| ---------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------- |
| 潜在                         | `[1,16,21,60,104]`（2,096,640 要素）                                                    | `text2video.py:159-166`、観測（計算） |
| DiT のグリッドとトークン数   | 21 × 30 × 52 = 32,760（seq_len の切り上げパディングなし）                               | 同上、`model.py:534-543`              |
| self-attn logits             | 1 head あたり 32,760² = 1,073,217,600 要素（f16 2.00 GiB）・全 12 head f32 で 47.98 GiB | 観測（計算）                          |
| cross-attn logits            | 12 × 32,760 × 512 = 201M 要素（f16 384 MiB）                                            | 観測（計算）                          |
| 残差流 `[1,32760,1536]`      | f16 96 MiB / f32 192 MiB                                                                | 観測（計算）                          |
| FFN 中間 `[1,32760,8960]`    | 293,529,600 要素・f16 560 MiB / f32 1,119.7 MiB（1.093 GiB）                            | 観測（計算）                          |
| DiT 1 forward                | 141.5 T MAC ≈ 283 TFLOP（Linear 40.96・self-attn 98.91・cross 1.55 T MAC）              | 観測（計算・norm と活性化は除く）     |
| 生成 1 本                    | 283 TFLOP × CFG 2 × 50 ステップ ≈ 28.3 PFLOP                                            | 同上                                  |
| VAE 出力                     | `[1,3,81,480,832]`（97M 要素・f32 370 MiB / uint8 92.5 MiB）                            | 観測（計算）                          |
| VAE チャンク内の最大テンソル | up2 の nearest ×2 出力 `[4,192,480,832]` = 306.7M 要素・f32 1,170 MiB / f16 585 MiB     | 観測（計算）                          |
| feat_cache 合計              | 944,286,720 要素・f32 3.6 GiB / f16 1.8 GiB                                             | 観測（計算）                          |
| チャンク分割しない場合       | 96ch 段の入力だけで f32 11.64 GiB                                                       | 観測（計算）                          |
| VAE デコード全体             | 4 フレームのチャンク 1 回で約 6.77 T MAC、全体で約 274 TFLOP（DiT 1 forward と同規模）  | 観測（計算）                          |

- self-attn は DiT 1 forward の約 70% を占める。
- VAE の 2 番目に大きいテンソルは up3 の conv3d 入力で、パディングと cache 連結込みの
  `[96,6,482,834]` = 231.5M 要素（f32 883 MiB）。カーネル側で暗黙にパディングすれば出力
  `[96,4,480,832]` の f32 585 MiB まで減る（観測: 計算）。
- 計算はすべて深掘りレッグのスクリプト（scratchpad `wan/calc.js`）の出力。公表値ではない。

### 3.10 公開性能値・ライセンス・配布先・参照実装

- **性能値**: RTX 4090・1 GPU・480P で 261.4 s / ピーク 8.19 GB。条件は `--offload_model True
  --t5_cpu`（T5 は CPU、DiT はステップ後に CPU へ退避）。8.19 GB は T5 を GPU に載せない前提の数字
  （HF `assets/comp_effic.png` の画像を読んだ観測、README `:17`・`:596-601`、arXiv 2503.20314 の
  abstract）。複数 GPU では 2 / 4 / 8 枚で 188.5 / 120.8 / 112.3 s。README 本文は「about 4 minutes」。
- **ライセンス**: Apache 2.0（README `:663-664`、HF cardData `license: apache-2.0`、sha
  37ec5126・lastModified 2025-03-01）。`google/umt5-xxl` トークナイザのライセンス表記は個別に確認して
  いない。
- **配布先**: HF `Wan-AI/Wan2.1-T2V-1.3B`（公式形式）、`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（全部品
  safetensors）、ModelScope `Wan-AI/Wan2.1-T2V-1.3B`。
- **参照実装（公式）**: `WanModel`（`wan/modules/model.py`: WanSelfAttention / WanT2VCrossAttention /
  WanAttentionBlock / Head）、`WanVAE`（`wan/modules/vae.py`: CausalConv3d / RMS_norm / Resample /
  ResidualBlock / AttentionBlock / Decoder3d）、`T5EncoderModel` と `umt5_xxl`（`wan/modules/t5.py`）、
  パイプライン `WanT2V`（`wan/text2video.py`）、CLI `generate.py --task t2v-1.3B`。
- **参照実装（diffusers）**: `WanPipeline` = { `WanTransformer3DModel`, `AutoencoderKLWan`,
  `UMT5EncoderModel`（transformers）, `T5TokenizerFast`, `UniPCMultistepScheduler` }（HF
  `model_index.json`）。
- **export の切り出し元**: 公式 `WanModel` は torch.export に向かない（`model.py:493-582`）。理由は
  4 つ。forward が Python リストを受ける、`grid_sizes.tolist()` が値依存（`:51`）、`rope_apply` が
  サンプルごとの Python ループで float64 complex を使う、`flash_attention` を直接呼び CUDA と
  flash_attn の存在を assert する（`attention.py:54`・`112`）。diffusers の
  `WanTransformer3DModel`（テンソル入力・attention は SDPA 経由）を切り出し元にするのが前提になる。

### 3.11 代替候補と蒸留系（参考）

- **Wan2.2 TI2V-5B**: dim 3072・24 heads・30 層・in/out 48・DiT は fp32 で約 5.0B（約 20.0 GB）。
  VAE は 4×16×16 圧縮・48ch（`Wan2.2_VAE.pth` 2.82 GB）。既定 1280×704・121 フレーム・24 fps で
  27,280 トークン（導出）。T5 は同じ umT5-XXL。Apache 2.0（Wan-Video/Wan2.2
  `wan/configs/wan_ti2v_5B.py:15-36`、README）。トークン数は 1.3B 480P よりやや少ないが、DiT 単体は
  約 3.5 倍で、新しい VAE も要る。
  - README は「24GB VRAM（RTX 4090）で 5 秒の 720P を 9 分未満」とする（`--offload_model True
    --convert_model_dtype --t5_cpu`）。diffusers は同じ `WanPipeline` の `expand_timesteps=True` で扱う
    （`pipeline_wan.py:140`）。
- **Self Forcing**（arXiv 2506.08009）: Wan2.1-T2V-1.3B を causal attention + KV cache の自己回帰型へ
  蒸留したもの。4 ステップ（[1000,750,500,250]）、潜在 3 フレームずつ生成、832×480 で単一 H100
  17.0 FPS（論文、WebFetch 要約経由で逐語性は中程度）。推論時に CFG を使うかは確認できていない。
  重みの配布形とライセンスは未確認。
  - §7.3 の「H100 で約 16 FPS」はプロジェクトページの値。17.0 FPS（論文）とは出典が違う 2 値で、
    測定条件の違いは突き合わせていない。
- **FastWan2.1-1.3B**・TeaCache などの速度値は §7.3。

## 4. karume の受け入れ状況

### 4.1 足りるもの（DiT 側）

- IR の op 語彙は 61 個（観測: `packages/runtime/tests/fixtures/op-contracts.json` の件数。
  `tools/exporter/README.md:500-503` によれば op 数の正本は適合表）。
- linear・layer_norm（affine 無しは exporter が ones/zeros を合成・`tools/exporter/README.md:600-610`
  付近）・rms_norm（`packages/runtime/src/ops/names.ts:29-33`・`210-211`）・gelu / gelu_tanh・sin は
  ある。rms_norm の weight は正規化軸長の rank-1 で、weight が rank-2 の形や weight 無しの直書きには
  irodori recipe のパッチの前例がある（`tools/export-recipes/irodori/patch.py:17-27`）。SiLU は公開
  op ではなく、融合ルール silu（sigmoid→mul）が供給する（`packages/runtime/src/runtime/fusion.ts:47`）。
- 融合 attention は q `[B,H,M,D]` と k/v `[B,Hkv,N,D]` で M≠N を受けるので、cross-attention は契約上
  表せる（`names.ts:216-243`）。Anima の DiT は attention を保存して使っている
  （`tools/export-recipes/anima/export.py:636-641`）。
- AdaLN は融合ルール adaln（layer_norm → reshape → `add(scale, one)` → mul → `add(shift)`）が供給する
  （`packages/runtime/src/runtime/fusion-rules/adaln.ts:26-49`）。変調ベクトルは `[1,…,1,dim]` の
  broadcast 形・入力順は実測形に固定。
- 時刻埋め込みの sinusoidal は、Anima ではホスト計算のグラフ入力 `timesteps_proj` として渡し、torch
  とのビット一致は諦めて atol 6e-7 で固定している（`packages/models/src/anima/sampler.ts:118-136`、
  `tools/export-recipes/anima/patch.py:24-26`）。Wan の float64 計算も同じ手筋でホストへ出せる
  （推測）。
- Anima の DiT は既に t/h/w の 3 軸 RoPE を使っている（`packages/models/src/anima/rope-base.ts:22-25`）。
  ただし表の組み立ては F'=1 専用で、t 位置は常に 0（`packages/models/src/anima/dit-tokens.ts:179-215`、
  205 行目）。
- DiT 側で新しい op は要らない見込み（推測・深掘りレッグ）。
  - patch_embedding の Conv3d(16→1536, kernel = stride = (1,2,2)) は窓が重ならないので、reshape で
    `[21,30,52,64]` にして Linear(64→1536) と同じ計算になる（導出・`model.py:456-457`）。
  - unpatchify の rank 7〜8 の reshape / permute は、rank 4 以下への書き直しかホスト化で済む
    （推測）。
- ホスト側処理（patchify / unpatchify / rope 表 / timestep 埋め込み / CFG）はすべて Anima でホストに
  出ている（`packages/models/src/anima/pipeline.ts:1-46`）。グラフ入口を patchify の後ろへずらせば、
  次元言語はトークン長 S の 1 シンボルで済む（ADR 0034 の手筋・`docs/ir-v2.md:270-284`）。

### 4.2 Anima の雛形（4 段・常駐）

- 段は text_encoder（Qwen3）→ text_conditioner → transformer（S 形 DiT・CFG と更新はホスト）→
  vae_decoder（常時タイル）（`packages/models/src/anima/pipeline.ts:1-46`・`141-146`）。
- 既定の residency `"per-stage"` は段ごとに Session を張っては畳み、VRAM の前提を「最大の段 1 本ぶん」
  に保つ。opt-in の `"transformer"` は DiT だけを generate を跨いで常駐させる
  （`pipeline.ts:1268-1306`、`docs/limitations.md:916-955`、ADR 0112）。
  - 常駐中は次の段の前に `fitsHeadroom` の試し確保で空きを量り、足りなければ先に退避する。OOM を
    踏んだら 1 回だけやり直す。Metal では試し確保が常に「入る」と答えるので退避は働かない。
- CFG は正と負の 2 本を同じ Session で順に回す（B=1 × 2 回・`pipeline.ts:526-538`）。
- 動画へ流用できるのは「段ごとの Session 開閉」と「形を固定したタイル decoder を、ホストで切り出し・
  ブレンド・貼り付けする」骨格（`packages/models/src/anima/tiling.ts:1-40`）。VAE グラフ自体は
  T=1 前提（入力は固定タイル `[1,16,64,64]`・rank 4 化済み）で、そのまま使えない。

### 4.3 足りないもの

- **conv3d**: 語彙・カーネル・exporter のどこにも無い（観測: `rg conv3d` が packages/ と
  tools/exporter/src でヒット 0。`quantize.py:11` のコメントだけ）。conv 系は conv1d / conv2d /
  conv_transpose1d / deform_conv2d の 4 本（`names.ts:262-266`・`281-283`・`303`）。
  - 台帳は「3D・動画専用（5）」を「実装しなくてよい」側に置き、その上で「2D 版が固まってから同型で
    足せる」と書く（`docs/op-vocabulary.md:24`・`42`）。
  - exporter の `_h_conv2d` は rank 4 以外を拒否し（`tools/exporter/src/karume/aten_handlers.py:1049-1062`）、
    conv3d を含む IR は verify が拒否する（`tools/exporter/tests/test_verify.py:1154-1176`）。
  - Anima の VAE パッチは CausalConv3d を conv2d に置き換えるが、成り立つのは T=1・時間 stride 1・
    `2·pad_t == kt−1` のときだけ。feat_cache が渡されると fail loudly し、`time_conv` は T=1 では実行
    されない経路として扱う（`tools/export-recipes/anima/patch.py:29-31`・`72-100`・`147-150`）。
- **rank ≤ 4**: strided コピー族（permute / expand / slice / cat / sym_prefix_slice / masked_fill の
  mask）と conv2d・attention・upsample の契約に掛かる（`packages/runtime/src/codegen/strided.ts:28-29`、
  `docs/limitations.md:708-716`）。elementwise は rank の上限が無い
  （`packages/runtime/src/codegen/elementwise.ts:111`）。rank ≥ 5 は exporter の正規化 3 パスが潰し、
  潰せない形は export 時に fail loudly（`tools/exporter/src/karume/normalize.py:783`・`870`・`995`）。
  動画 VAE の `[B,C,T,H,W]` はこの上限に当たる。
  - Wan VAE の upsample3d は `time_conv` の出力を `reshape(b,2,c,t,h,w)` → `stack(dim=3)` → `reshape`
    で時間方向にインターリーブする。途中の形は rank 6（`vae.py:134-141`）。
  - mid の単一 head attention は SDPA で、各フレーム 6240 トークン × 384 次元（logits 38.9M 要素）
    （`vae.py:223-262`）。フレームは `rearrange(b c t h w → (b t) c h w)` でバッチ軸へ畳んでから回す。
    karume の融合 attention の契約に乗るかは未確認。
- **pad の制約**: pad は「最終次元・定数 0」だけ（`names.ts:156-163`、`docs/limitations.md:718-735`）。
  CausalConv3d の時間軸先頭へのゼロ詰めは書けず、conv の padding 引数へ畳むか cat で表す必要がある。
- **upsample**: `upsample_bilinear2d`（align_corners=True 専業）と融合ルール upsample2x（f32 rank4
  NCHW の空間 2 倍）だけで、時間軸の upsample は無い（`names.ts:305-319`、
  `packages/runtime/src/runtime/fusion-rules/upsample2x.ts:35`）。upsample2x は NCHW の最終 2 軸を
  行に畳むので、時間をバッチに置いた `[T,C,H,W]` には乗る（推測・`kernels/upsample2x.ts:1-12`）。
- **feat_cache の持ち越し**: 時間方向の因果 cache を run 間で持ち越す仕組みは IR に無い。state
  スロット（ADR 0066）と `state_append` は attention 用の `[B,Hkv,M,D]` rank-4 専用
  （`names.ts:244-259`、`packages/runtime/src/ops/shapes.ts:883`、`format/ir.ts:164`）。
- **complex 形 RoPE**: 専用 op は無く、cos/sin 表はホストからのグラフ入力、適用はグラフ内の
  elementwise（`packages/runtime/src/runtime/fusion-rules/rope.ts:35-49`）。融合ルール rope と
  カーネルは half-split 専用で、interleave 形は意図的に受理しない（`kernels/rope.ts:1-8`・`31-33`）。
  - complex 形には irodori recipe の前例がある。`view_as_complex` を rank 4 のまま実数対の入れ替え +
    cos/sin の要素積へ書き換える（`tools/export-recipes/irodori/patch.py:12-16`・`131-176`）。
  - （推測）Wan も同じ実数化パッチで正しく動くが、rope 融合に乗らず非融合の primitive 列で走る。
    q/k 射影の出力チャネルを並べ替えて half-split へ寄せる手が eager 同値で成り立つかは未確認。
  - 複素数 dtype は無い（意味論 dtype は f32 / i32 / bool・`tools/exporter/README.md:505-506`）。cos は
    「実測に出るまで足さない」とされている（`names.ts:22-25`）。
- **adaLN 融合**: Wan の `norm(x)*(1+scale)+shift` を export したノード列が adaln ルールの固定順と
  一致するかは未確認。（推測）式の順序が違えば融合に乗らず非融合で動く。
- **T5 の相対位置バイアスと mask 契約**: 融合 attention の加算 mask は f32・rank 4・ちょうど
  `[1,1,M,N]` だけで、`[1,H,M,N]` は受理しない（`names.ts:226-229`、`shapes.ts:866`）。umT5 の
  head ごとのバイアス `[1,64,L,L]` は融合 attention に入らない。
  - （推測）SDPA の保存はターゲット別の opt-in なので、保存しない分解経路（bmm → add → softmax →
    bmm）なら新しい op 無しで通る（`tools/exporter/src/karume/convert.py:299-310`）。rowBlockAttention
    の適用条件に合うかは未確認（`fusion-rules/row-block-attention.ts:96-100`）。
  - （推測）バイアスを export 時に const へ焼くと、1 層ぶん（H=64・f32）で 64 MiB になり、const の
    32 MiB 上限（§4.4）を超えうる。
- **T5 系列の不在**: T5 / umT5 の encoder を動かす系列は無い（観測: `rg -i
  'umt5|t5encoder|T5EncoderModel|t5_encoder|relative_attention_bias|T5LayerNorm'` がリポ全体で 0 件）。
  Anima の T5 は Unigram トークナイザの id 列を conditioner に渡すだけ。T5 トークナイザの TS 実装
  （SPM 正規化・Unigram）はある（`packages/models/src/anima/text/t5-tokenizer.ts`・`spm-normalizer.ts`）。
- **画像 1 枚前提の形状**: `ditPatchGeometry` は pt=1・正方 patch を仮定し、patchify / unpatchify は
  空間 2 軸だけ（`packages/models/src/anima/dit-tokens.ts:1-14`・`44-61`・`105-120`）。トークン長の
  上限は `MAX_DIT_TOKENS = 16384`（`packages/models/src/anima/resolution.ts:66-70`）と
  `torch.export.Dim("S", max=16384)`（`tools/export-recipes/anima/export.py:267-272`・`568`）。
  Wan の 32,760 はこれまで扱った最大 S の約 2 倍。

### 4.4 容量

- **融合 attention は S（スコア行列）を実体化する** 3 dispatch 方式（QK → 行統計 → PV）。online softmax
  への書き換えは、ビット同一の門を失うため MUST NOT（`packages/runtime/src/kernels/attention.ts:1-40`）。
  - レッグ間の食い違い: Wan の op 棚卸しレッグは「logits を全体では実体化できないので、flash 型
    （タイル分割の online softmax）が必須」と書いた（Wan 掃引・先行事例レッグも flash 型を前提に置く）。
    karume の受け入れレッグは上のとおり「S を行ブロックで実体化・online softmax は MUST NOT」とする。
    本記録の容量の見積りは後者で書いた。後者を優先するのは karume の既存の判断（ビット同一の門）に
    従ったからで、動画の長系列のためにこれを見直すかは別の裁定（本記録は裁定しない）。
  - S 1 枚 = B·H·block·N·格納幅 が `maxStorageBufferBindingSize` に収まるよう、`planRowBlocks` が
    クエリ行を等分する。1 行でも上限を超えれば fail loudly
    （`packages/runtime/src/runtime/recipe-builders/attention.ts:166-176`・`211-216`・`279-290`、
    `fusion-rules/row-block-attention.ts:55-94`）。
  - 試算（観測: node で計算・H=12・M=N=32,760）: 1 行あたり f16 格納 786,240 B・f32 1,572,480 B。

| 束縛上限                            | 行ブロック数（f16 格納 / f32 格納） |
| ----------------------------------- | ----------------------------------- |
| 2 GiB−4（RTX 3080 Ti・Vulkan など） | 12 / 24                             |
| 4 GiB−4（M2）                       | 6 / 12                              |
| 128 MiB（WebGPU 仕様の既定）        | 193 / 386                           |

- **device limits**: runtime はアダプタの最大値を requiredLimits に要求する（`maxBufferSize` を先に
  絞り、`maxStorageBufferBindingSize` をその値で clamp・`packages/runtime/src/gpu/acquire.ts:52-64`・
  `83-102`）。重みと中間は 1 本ずつ束縛上限と比べ、確保の前に全件列挙して落とす
  （`runtime/weight-residency.ts:412-480`、`runtime/transient-plan.ts:257-266`）。合計が物理 VRAM を
  超えるかは検査しない（Metal では errorScope も沈黙）。
  - 実測値: RTX 3080 Ti（Vulkan）は `maxStorageBufferBindingSize` 2 GiB−4・`maxBufferSize` 1 TiB、M2 は
    4,294,967,292 / 14,302,248,960（`docs/limitations.md:1570-1584`、
    `docs/research/2026-08-02-anima-recon.md:146-150`）。D3D12 は 2 GiB 固定とコードの注記にある
    （`recipe-builders/attention.ts:171`）。
- **FFN 中間** `[32760,8960]` f32 = 1,174,118,400 B（1.093 GiB）は 2 GiB−4 に収まる。CFG を B=2 に
  まとめると 2,348,236,800 B で上限を超えるので、Anima と同じ B=1 の 2 回 forward が前提になる
  （観測: 計算）。
- **B=1 前提**: Anima の CFG（B=1 × 2 回）、融合 attention の mask `[1,1,M,N]`、EmbeddingGemma の
  batch>1 export の既知の不具合（`docs/known-issues.md:266-286`）、台帳に未消化の「conv 契約拡張
  （B=1 等の緩和）」（`docs/op-vocabulary.md:386`）。conv1d / conv2d の recipe はバッチを z 軸に置く
  作りで、B>1 は構造上通る（`recipe-builders/conv.ts:226-227`・`351-352`）。
- **const の 32 MiB 上限**: 配布形の block 上限は 32 MiB。1 行がこれを超えるテンソルと、32 MiB を
  超える const は配布できない（書き手が fail loudly・逃げ道は未決・`docs/limitations.md:1347-1357`、
  `docs/container-v1.md:271-296`・`754-783`）。
- **Deno の VRAM 97%**: Deno では GPUBuffer の総確保がドライバ申告予算の 97% で頭打ちになり、天井付近
  では OOM ではなく device lost になる（外部制約・`docs/limitations.md:309-360`）。
- **その他**: Chromium は単一 ArrayBuffer を 2,145,386,496 B で打ち切る（分割形の容器なら影響なし・
  `docs/limitations.md:1331-1345`）。静的形状前提（`docs/decisions/0004-execution-model.md:22-24`）。
- **i64 境界**: IR に i64 は無く、exporter 境界で値域検査つきの i32 へ正規化する（ADR 0009・
  `docs/decisions/0009-dtype-i32-bool.md:15-17`）。emit された i32 演算の中間値は 2³¹ を跨ぐとラップ
  する（`docs/limitations.md:672-681`）。（推測）動画の添字（トークン添字は 32,760 程度）は i32 の
  範囲内で、ホストへ出した添字計算は TS の number なので問題の外にある。

### 4.5 計算量と dispatch

- 試算（観測: node で計算・未実測）: QK と PV だけで 1 層あたり約 6.6 TFLOP、30 層で約 198 TFLOP /
  forward。Linear 系を足した 1 forward は約 283 TFLOP（§3.9）。行ブロック分割で容量は収まるが、
  計算量は S² のまま残る。
- GEMM 系（linear / attention の ①③ / conv の implicit GEMM）は 1 workgroup = 1 タイルで、
  `maxComputeWorkgroupsPerDimension`（65,535）を超えると fail loudly（`recipe-builders/attention.ts:195-198`、
  `docs/limitations.md:225-240`）。conv2d は Hout·Wout > 8,388,480（M2 プロファイルでは 4,194,240）で
  落ちる。832×480 の 1 フレーム（399,360）は収まる。
- submit は時間予算でチャンクに分割する（TDR 2 秒 / Chromium watchdog 対策）が、分割の単位は
  dispatch 間なので、1 dispatch の重さは分割できない（`docs/decisions/0004-execution-model.md:27-43`）。
  - （推測・試算）Wan の FFN 上り 1 本は 32760×1536×8960×2 ≈ 0.90 TFLOP。遅い GPU では単発 dispatch が
    TDR 予算に近づきうる（未実測）。

## 5. 無い op の追加経路と規模の物差し

### 5.1 入場門の判定材料

- op 追加の手順は ADR 0065 ではなく、`docs/op-vocabulary.md` の「入場門モデル」（ADR 0059 / 0064）に
  ある（`docs/op-vocabulary.md:128-186`・`352-368`）。安い順に、export 消滅層（定数畳み込み・分解）→
  Core ATen 層（Tag.core 実測・台帳 NOTE のみ）→ 拡張原子 / 分子層（非 core・ADR 必須）→ 融合層。
  ADR 0065 が定めるのは「PyPI karume = 汎用 core・モデル別 recipe は `tools/export-recipes/`・依存は
  recipe → core の一方向」という境界（`docs/decisions/0065-exporter-core-recipe-split.md:25-50`）。
- 1 op 足すときに揃える契約は OP_CONTRACTS + `karume/ops.py` + `shapes.py` +
  `fixtures/op-contracts.json` + CPU 参照 + golden COVERAGE の 1 セット
  （`docs/decisions/0059-op-vocabulary-entry-doors.md:70-72`）。超越関数を含む op には数値危険クラスの
  門が MUST。
- Core ATen の帰属（観測: `tools/.venv/bin/python`・torch 2.13.0+cpu で `torch.Tag.core in op.tags`）:

| Core ATen 外                                                                                          | Core ATen 内                                                                                                                       |
| ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| conv3d・conv2d・conv1d・upsample_nearest3d・upsample_trilinear3d・view_as_complex・view_as_real・SDPA | aten.convolution・avg_pool3d・max_pool3d_with_indices・_adaptive_avg_pool3d・replication_pad3d・native_group_norm・constant_pad_nd |

### 5.2 conv3d の 2 案

|          | 案 A: IR に op `conv3d` を足す                                                                    | 案 B: recipe で conv2d に分解する                                                         |
| -------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 中身     | `aten.conv3d.` を `PRESERVED_OP_PREFIXES` に足して保存し、直接カーネル + implicit GEMM を足す     | `conv3d = Σ_kt conv2d(slice(x, 時間 kt…), W[:,:,kt])`（時間軸を conv2d のバッチ軸に置く） |
| 入場門   | 拡張分子層 = ADR 必須の公算が高い（同じく Core ATen 外の conv2d が ADR 0059 決定 5 で拡張分子層） | 新しい op を足さない（recipe の patch 層の担当）                                          |
| 中間容量 | implicit GEMM なら中間ゼロ                                                                        | 出力サイズの中間が kt = 3 倍 + slice の実体化コピーと add                                 |
| 前例     | conv1d の implicit GEMM が「2D 版の 1 軸版」として同型で書かれている                              | Anima の T=1 パッチ（T>1 には効かない）                                                   |

- 案 A の根拠（`docs/decisions/0059-op-vocabulary-entry-doors.md:28-35`・`51-60`、
  `tools/exporter/src/karume/convert.py:285-297`）: conv 系は保存しないと汎用の `aten.convolution`
  形になる（`convert.py:273`）。core の `aten.convolution`（空間 3 軸）の attr 変種を別 op 名として
  扱えば Core ATen 層（台帳 NOTE のみ）という読みも成り立ちうるが、前例は無い（推測）。
- 案 A の実装面: conv2d の implicit GEMM（ADR 0024）は A タイル（重み `[Cout,K]` の平坦化）が 1D / 2D
  で完全に共通で、重み `[Cout,Cin,Kt,Kh,Kw]` も平坦化すれば同じ形になる
  （`packages/runtime/src/kernels/gemm.ts:1851-1856`）。次元に依存するのは 4 か所だけ。
  - uniform の幾何の語（`CONV2D_DIMS_EXTRA` の 12 語・`gemm.ts:1773-1793`）
  - x の暗黙 gather（xcol・`gemm.ts:1825-1838`）
  - 平坦 k の分解（`conv2dKDecode`。3D では (ic, kt, kh, kw)・`gemm.ts:1948-1987`）
  - B タイルの充填（`fillBConv2d`）
  - （推測）n から (ot, oy, ox) を復元するのに height_out の語が要り、uniform は {m,n,k} の 3 語 + 幾何
    18 語 = 21 語で 24 語の確保になる。v4 判定（`kFlat%4`・`Wout%4`・`strideW==1`・
    `kernels/conv2d.ts:90-103`）とビット同一の土台（平坦 k の昇順・bias-first・範囲外は 0）は 3D でも
    同じ規律で保てる。
  - conv1d の implicit GEMM（`gemm.ts:2080-2245`）が DIMS_EXTRA・XCOL・kDecode・fillB を次元違いで
    複製した前例で、3D 版も同じ型で書ける。
- 案 B の不利: 前例ガイドは「中間 1.5〜2 倍の時点で既に保存側の前例がある」と書く
  （`docs/op-vocabulary.md:150-159`、ADR 0059 `:44-50`）。kt = 3 倍の案 B はこれに照らして案 A 側の
  根拠になる。なお `(3,1,1)` の `time_conv` は `[1,C,T,H·W]` 上の `(3,1)` カーネルの conv2d と厳密に
  同値（推測・導出）。
- im2col の実体化は conv2d で単一バッファ 3.62 GB として ADR 0024 で却下済み
  （`docs/decisions/0024-conv2d-implicit-gemm.md:16-18`）。3D で時間方向をチャネルへ kt 枚 cat する
  分解は入力の kt 倍の中間になり、Wan VAE の最高解像度段で約 1.84 GB と見積もられた（推測・記憶
  ベースの形状による試算で、§3.9 の一次ソース由来の形とは突き合わせていない）。

### 5.3 conv3d の周辺で決まっていない点

- **因果パディング**（推測の代案 3 つ）: feat_cache（前チャンクの末尾 2 フレーム）をグラフの入出力と
  して受け渡し、初回はホストがゼロを渡す（結合は rank 4 の cat）／conv3d の attrs に非対称の時間
  padding を持たせる／ゼロ定数を cat する（数百 MB の定数になり非推奨）
  （`tools/export-recipes/anima/patch.py:75`）。
- **rank**: B=1 を落として `[C,T,H,W]` の rank 4 で回すなら、5D の conv3d は reshape（コピー無しの
  別名）で挟むか、unbatched の rank-4 入力を契約にする（後者は推測）
  （`packages/runtime/src/ops/shapes.ts:100-117`・`487-488`・`515-516`、`normalize.py:50-52`・`780-787`）。
  `time_conv` の出力の時間インターリーブは rank 6 を経由する（§4.3）ので、これも rank 4 以下への
  書き直しが要る。
- **VAE の export 形**（深掘りレッグの提案・`vae.py:101-160`・`475-480`・`582-589`）: 公式・diffusers
  とも feat_cache を Python のリストと番兵（`'Rep'` 文字列 / `first_chunk` フラグ）で管理していて、
  そのままでは静的なグラフにならない。
  - 提案は、潜在 1 フレームと cache テンソル群を入力にし、出力フレームと更新後の cache を返す 1 チャンク
    分のグラフに組み替え、最初のチャンク用とそれ以降用の 2 種類を export する形。
  - cache が要るのは時間カーネル 3 の CausalConv3d だけ（shortcut の 1×1×1 conv は使わない）。
  - diffusers には `tiled_decode`（tile 256・stride 192 px）もあるが、公式にタイル分割は無い。
- **dispatch 上限**: conv2d の implicit GEMM は workgroups = [ceil(n/tileN), ceil(m/tileM), batch] で、
  上限を超えたら fail loudly（`recipe-builders/conv.ts:390-394`、`codegen/dispatch.ts:19-36`）。2048px
  の VAE で n タイルが 65,536 になり上限を 1 超えた前例がある（ADR 0024 決定 7・`:37-41`）。conv3d で
  n = Tout·Hout·Wout にするとフレーム数で上限に近づく。時間軸を z 軸（バッチ）に置けば n は H·W の
  ままで済む（推測）。
- **添字**: カーネルの添字と uniform はすべて u32 で、値の検査は `assertU32Params` の 1 か所
  （`packages/runtime/src/codegen/params.ts:12-30`）。conv3d 自体は整数テンソルを扱わないので i64
  境界の影響は無い。
- **geometry profile**: conv2d の implicit GEMM の幾何は adapter の geometry profile から引き、キー・
  WGSL・dispatch へ同じ値を通す MUST がある（`recipe-builders/conv.ts:372-379`）。conv3d でプロファイル
  の欄を共有するか新設するかの判断が要る。新設なら tune の掃引ケースにも波及する。調査時点では
  tune 層（`packages/runtime/src/tune/derive.ts`）が ADR 0117 段 3 で作業ツリー上の改修中だったが、
  その後 a71f51a9 / 6d0fae81 で段 3 は記録済み（観測: `git log`）。

### 5.4 規模の物差し（git の実測）

| コミット | 内容                                                          | 規模（numstat・バイナリ除外）                                                                                        |
| -------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 7a725d7f | conv1d(groups==1) を implicit GEMM へ置換                     | 17 ファイル +6371/−78。src 3 ファイル +403・テスト 2 ファイル +950・WGSL スナップショット 12 本 +5018                |
| 947298d0 | upsample_bilinear2d と deform_conv2d を語彙へ追加（ADR 0055） | 32 ファイル +3297/−48。src 11 ファイル +1312・テスト 9 ファイル +1333・fixture 3 ファイル +335・docs 5 ファイル +317 |
| de8bae93 | sin を足しただけ                                              | 15 ファイル +116/−12。src 6 ファイル +26                                                                             |
| cd25abf5 | gru_scan（ADR 0056）                                          | 49 ファイル +3417/−769                                                                                               |

- 案 A の見積り（推測）: 手書きの src 約 1,000 行（runtime 約 800・exporter 約 150）、テスト約
  1,500 行、WGSL スナップショット 12〜16 本、ADR 1 本（約 150〜200 行）。i4 は conv2d と同じく対象外
  とする前提。根拠は conv2d の現行の足跡（`kernels/conv2d.ts` 383 行・`gemm.ts` の conv2d 断片 約 180
  行・parity テスト 801 行・スナップショット 16 本）に 3 軸目の増分を足したものと、上表の 1 op あたり
  の規模。
- 触るファイル（conv2d の足跡から列挙・`rg -l conv2d` は runtime src 24・runtime テスト 31・exporter
  15 ファイル）:
  - exporter: `convert.py`（PRESERVED_OP_PREFIXES）・`aten_handlers.py`（conv2d 版は `1027-1087`）・
    `ops.py`・`shapes.py`（conv2d 版は `1091-1147`）・`golden_models.py` / `goldens.py`・`quantize.py`
  - runtime: `ops/names.ts`（WEIGHT_SLOTS / WEIGHT_CHANNEL_AXES）・`ops/attrs.ts`（conv2d 版は
    `702-756`）・`ops/contracts.ts`・`ops/shapes.ts`（conv2d は `674-730`）・`reference/ops.ts`（conv2d
    は `1230-1299`）・`kernels/conv3d.ts`（新規）・`kernels/gemm.ts`・`runtime/recipe-builders/conv.ts`・
    `runtime/recipe-builder.ts:604`・`runtime/plan.ts`・`kernels/geometry-profile.ts`
  - テスト: `fixtures/op-contracts.json`・ops_contract / reference_ops / codegen_wgsl（スナップショット）
    / gpu_ops / gpu_full_write / gpu_gridstride / gpu_conv3d_parity（新規）/ e2e_golden、exporter の
    test_aten_handlers / test_convert / test_goldens
  - docs: ADR と `docs/op-vocabulary.md` の NOTE

## 6. テキストエンコーダの選択肢

前提（§3.7・§4.3）: umT5-XXL は 5.68B パラメータ・512 トークンで約 2.42 T MAC。karume に T5 系列は
無く、相対位置バイアスは融合 attention の mask 契約に入らない。公式自身は `--t5_cpu` で T5 を CPU に
逃がしており、公開性能値 8.19 GB も T5 を GPU に載せない数字（§3.10）。

| 方式                                  | 守るもの                                                                                  | 失うもの・リスク                                                                                                                             |
| ------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) GPU で i8 / i4 に量子化して載せる | 全段がブラウザ内で閉じる・任意プロンプト                                                  | 重みの DL（推測: i8 で約 5.7 GB・i4 で約 3 GB 前後）・export のホスト RAM・量子化品質（下記）・f16 活性の溢れ（下記）・T5 系列の新規実装     |
| (b) GPU で f16 のまま載せる           | 量子化誤差が無い                                                                          | 約 11.4 GB の DL と VRAM・fp16 で NaN の報告がある（下記）・WebGPU に bf16 は無い                                                            |
| (c) 事前計算した埋め込みを配る        | T5 の実装と数値リスクがゼロ・DL は 1 プロンプト数 MiB（推測: `[512,4096]` bf16 で 4 MiB） | 任意プロンプトを受けられない（固定プロンプト集のデモに限る）                                                                                 |
| (d) CPU で回す                        | 公式 `--t5_cpu` と同じ構成・VRAM を使わない                                               | （推測）karume の CPU 側はテスト用の参照実装（`packages/runtime/src/reference/`）しか無く、実用速度の経路は無い・ブラウザ CPU での所要は未測 |

- **量子化の前例**: 格納型は f32 / f16 / i8（per-channel）/ i4（group）/ i2（gemma4-qat 限定）
  （`docs/quantization.md:18-28`・`86-93`）。i4 の実行経路は linear / embedding / conv1d の implicit
  GEMM（groups==1）だけ（`kernels/weight-storage.ts:42-44`）。テキスト長 512 級の encoder は GEMV 族
  （M ≤ 64）ではなく tiled GEMM 側に乗る（`kernels/linear-gemv.ts:1-6`・`165`）。数 B 級の前例は
  gemma4 E2B / gemma4-qat E2B・E4B の i4 系列だが、E4B 通常版の export はホスト RAM 48 GB 以上が要り、
  31 GiB 機では OOM する（`docs/backlog.md:239-241`）。（推測）umT5-XXL の export も同等以上の RAM が
  要る。
- **語彙埋め込み**: `256,384 × 4,096` は f32 で 4,200,595,456 B（2 GiB−4 を超える）、f16 で
  2,100,297,728 B（ぎりぎり収まる）、i8 で 1,050,148,864 B（観測: 計算）。WebGPU 仕様の既定上限
  128 MiB は大きく超えるが、参照するのは最大 512 行なので CPU 側の gather で回避できる（推測）。
- **量子化品質の報告**（一次・当該実装の自己計測・第三者の追試なし）: mlx-umt5 は MLX の group-affine
  量子化で出力が壊れ（cos ≈ 0.108）、per-channel 対称 int8 だけが安全と報告する。理由は「attention
  出力のスケーリングが無く、残差が 24 層で単調に増える」。int8 版はディスク約 6.3 GB・ロードピーク
  7.02 GB、bf16 と torch-fp32 の cos は 0.999575（512 トークン）
  （<https://github.com/sb1992/mlx-umt5>）。karume の i8 は per-channel、i4 は group（推測: i4 はこの
  報告の壊れ方に近い側）。
- **f16 実行の報告**: T5 系は fp16 で FFN の活性が 65504 を超える既知の問題があり、transformers は inf
  を検出して clamp する（<https://github.com/huggingface/transformers/issues/5651>、
  <https://github.com/huggingface/transformers/pull/10956>）。Wan の umT5 については、
  ComfyUI-WanVideoWrapper のメンテナが「fp16 では NaN が出て動画が壊れる」と述べ、PR 投稿者は NaN/Inf
  は見つからなかったと反論している（PR 未マージ・
  <https://github.com/kijai/ComfyUI-WanVideoWrapper/pull/1388>）。公式 t5.py の `fp16_clamp`（§3.7）も
  同じ懸念を示す。（推測）WebGPU の f16 で回すなら、活性は f32 にするか同じクランプを入れるかを実測で
  決める必要がある。
- **差し替え・事前計算の事例**: Wan 向けに umT5 を小さいエンコーダへ差し替えた事例は見つからない。
  T2I で T5-XXL を 50 分の 1 の蒸留 T5-base に置き換えた研究はある（Wan ではない・
  <https://arxiv.org/html/2503.19897v1>）。埋め込みの事前計算は学習用データセットの例だけ
  （`huiwon/droid_wan_umt5_cache`・検索スニペット）で、推論用に配布した事例は見つからない。
- **共通**: (a)〜(d) のどれでも、negative prompt の埋め込みは固定なのでキャッシュでき、有効長だけで
  回せる（導出・§3.7）。

## 7. 先行事例と相場

### 7.1 ブラウザ事例

- Wan2.1（全サイズ）をブラウザ・WebGPU・ONNX Runtime Web・wgpu で動かした公開事例は見つからない。
  ONNX へ export した事例も見つからない（観測: 検索 7 クエリ。ヒットは Gradio の Web UI である Wan2GP と
  一般的な WebGPU 解説のみ）。
- ブラウザ内拡散モデルの代表例は Web Stable Diffusion（MLC / TVM・2023・画像のみ）。検証機は Apple
  silicon のみ、GPU メモリ 8 GB 程度、当時は FP32 のみ。Chrome が配列アクセスに境界チェックを挿入
  するため Apple silicon で約 3 倍遅くなり、`--enable-dawn-features=disable_robustness` を求めていた
  （<https://github.com/mlc-ai/web-stable-diffusion>）。
- ONNX Runtime Web には protobuf 2 GB の制限と、wasm32 の 4 GB 制限による「4GB 超のモデルは現状動か
  せない」がある（<https://onnxruntime.ai/docs/tutorials/web/large-models.html>）。

### 7.2 ネイティブ非 CUDA 実装

- **stable-diffusion.cpp**（ggml・Vulkan / Metal）が Wan2.1 / 2.2 に対応。推奨は拡散モデル GGUF
  Q8_0・umT5 GGUF・VAE safetensors で、docs に「Wan models vae requires really much VRAM!」とある
  （<https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/wan.md>）。利用者報告
  （<https://github.com/leejet/stable-diffusion.cpp/discussions/1000>・`/868`）:
  - Vulkan/ROCm・1.3B FP16・384×216・85 フレーム・20 ステップで VAE ピーク約 5.4 GiB・システム RAM
    約 500 MiB。
  - RX 7600 XT 16 GB・416×240・8 秒で約 23 分（VAE は CPU へ退避）。
  - Wan2.2 の 832×480 は VAE バッファが 16 GB を超える（Wan2.2 は別の VAE）。
  - Snapdragon 865（Android）ではほぼ成功しない。
- **MLX 移植**: Blaizzy/mlx-video（Wan2.1 1.3B/14B・既定 50 steps / shift 5 / guide 5）と mlx-video-rs
  （Rust）。M2 Max 32 GB・1.3B 4bit の実測（二次・<https://note.com/mikai_daichi/n/nab2a5d452f83>）:
  512×512・9 フレーム・50 ステップで総 229.3 秒（T5 12.2 / 拡散 212.2 / VAE 4.5 秒）、256×256・9 フレーム
  で総 70.0 秒（T5 19.0 / 拡散 48.5 / VAE 1.3 秒）。32 GB 機では 4bit 化がメモリ上必要だったとある。
- **PyTorch MPS**（二次・<https://kennycason.com/posts/2025-05-20-wan2.1-on-macos.html>）: 832×480・
  48 フレーム・15 ステップ、確実に動いた上限は 32 フレーム、RAM 約 100 GB（M4 Max 128 GB）。総所要の
  記載なし。
- **Draw Things**（Metal）: 1.3B を f16 版と 8-bit 版で提供し「≥8GB RAM」とする。所要時間は非公表
  （<https://wiki.drawthings.ai/wiki/Video_Generation_Basics>）。Metal FlashAttention v2.5 の記事には、
  M5 iPad 16GiB で Wan 2.2 A14B による 448×768・5 秒の動画生成が可能という記述だけがある（時間の
  内訳は無い・<https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural>）。

### 7.3 速度・量子化の相場

- 公式: RTX 4090 で 261 s / 8.19 GB（§3.10）。A800 単機・832×480 でベースライン約 175 秒、TeaCache
  閾値 0.05 / 0.07 / 0.08 で約 117 / 110 / 88 秒
  （<https://raw.githubusercontent.com/ali-vilab/TeaCache/main/TeaCache4Wan2.1/README.md>）。
- 民生 GPU の異常値: RTX 4060 Laptop・832×480・81 フレーム・50 ステップで 424.46 s/it・総 4〜5 時間・
  VRAM 約 11 GB（<https://github.com/Wan-Video/Wan2.1/issues/555>・メンテナ回答なし・原因未確認）。
  「RTX 4060 8GB + GGUF Q4 で 4〜6 分」は二次記事の主張のみで一次の実測は見つからない。
- **蒸留系**: FastWan2.1-1.3B は H200 で 5 秒 480P が総 5 秒（ノイズ除去 1 秒）、RTX 4090 で総 21 秒
  （ノイズ除去 2.8 秒）（<https://haoailab.com/blogs/fastvideo_post_training/>）。差し引きで T5 と VAE が
  約 18 秒を占め、**蒸留後の律速は T5 / VAE へ移る**（推論）。Self-Forcing は初回遅延約 0.8 秒、
  ストリーミングが H100 で約 16 FPS・4090 で約 10 FPS、要件は 24 GB 以上の Nvidia GPU
  （<https://self-forcing.github.io/>）。H100 の値は、論文の 17.0 FPS（§3.11）とは出典が違う 2 値。
- **M2 無印の見積り**（推測）: 実測は見つからない。A800 の 175 秒から実効約 160 TFLOPS と整合する。
  M2 Max の MLX 実測（実効約 4.6 TFLOPS）を GPU コア数比（10/38）で縮めると、無印 M2 で 28.3 PFLOP は
  数時間規模。4 ステップ・CFG 無しの蒸留版なら約 25 分の 1。
- **配布サイズ**: 1.3B の GGUF は F16 2.84 GB / Q8_0 1.54 GB / Q6_K 1.2 GB / Q5_K_M 1.09 GB / Q4_K_M
  983 MB / Q4_0 866 MB / Q3_K_S 655 MB（<https://huggingface.co/samuelchristlie/Wan2.1-T2V-1.3B-GGUF>）。
  1.3B の fp8 版は Comfy-Org / Kijai には見当たらない。umT5 は fp16 11.4 GB・fp8_e4m3fn_scaled 6.74 GB・
  GGUF Q8_0 6.04 / Q5_K_M 4.15 / Q4_K_M 3.66 GB（作者は imatrix 無しと注記し Q5_K_M 以上を推奨・
  <https://huggingface.co/city96/umt5-xxl-encoder-gguf>）。
- **量子化品質**: 1.3B の weight-only 量子化の系統立った評価は見つからない。W/A 量子化の論文
  QuantSparse（480×832・80 フレーム・CFG 6）は FP16 VQA 73.12、Q-VDiT W4A8 で 56.45、QuantSparse W4A8 +
  注意密度 15% で PSNR 20.88（<https://arxiv.org/html/2509.23681>・WebFetch 要約経由で表は未逐語確認）。
  weight-only には直接使えない。
- Wan の技術レポートは FP8 GEMM で DiT 1.13 倍・8-bit FlashAttention で 1.27 倍超・diffusion cache で
  1.62 倍を挙げる（arXiv 2503.20314v2・WebFetch 要約）。fp8 は Metal / MPS で動かない（二次）。

### 7.4 WebGPU の制約 4 点

1. **バッファ上限**: 仕様の既定は `maxBufferSize` 268,435,456 B（256 MiB）・
   `maxStorageBufferBindingSize` 134,217,728 B（128 MiB）。実際はアダプタ依存で、requestDevice で明示
   的に要求する（<https://www.w3.org/TR/webgpu/>・curl で取得した本文を grep。WebFetch の要約は
   maxBufferSize を 2^31 と誤報していた）。二次記事では Chrome（M3 Pro）で両方 4,294,967,292 B。
2. **i64 / f64 が無い**: WGSL のスカラー型は bool・AbstractInt・AbstractFloat・i32・u32・f32・f16 のみ
   （<https://www.w3.org/TR/WGSL/>）。
3. **公式 Wan の float64 区間**: sinusoidal 時刻埋め込み（`model.py:18-28`）と RoPE（`:31-39`・`:55`）
   が float64、変調・残差・head は fp32 固定（`:297-313`・`:344-346`・`:546-548`）。WebGPU では f32 で
   組み直すかホストで事前計算する。f64 → f32 の誤差の報告は見つからない。
4. **TDR とウォッチドッグ**: Windows の TdrDelay は既定 2 秒・TdrLimitCount は 60 秒内 5 回
   （<https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys>）。Chrome の
   GPU プロセスにはドライバ操作が長引くと device lost にするウォッチドッグがあり約 10 秒（準一次・
   <https://toji.dev/webgpu-best-practices/device-loss.html>）。二次記事には「Chrome は 2 秒」とあり
   食い違う。仕様は「may set up "watchdog" timer」で実装任意。
   - （推測・試算）32,760 トークンの自己注意 1 層は約 6.6 TFLOP。実効 1〜5 TFLOPS の GPU で 1 dispatch
     に積むと 1.3〜6.6 秒で、TDR 2 秒を超えうる。WebGPU での長時間動画生成の device lost 報告は無い
     （先行事例が無い）。

### 7.5 VAE の省メモリ 3 系統

| 系統                   | 中身                                                                                                                                                                               | 根拠                                                                                   |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 公式の因果チャンク     | 潜在 1 フレームずつデコード + 2 フレーム分の feat_cache                                                                                                                            | `vae.py:14`・`544-567`、arXiv 2503.20314 §4.1                                          |
| diffusers の空間タイル | `enable_tiling`（v0.40.0）。256×256 px・ストライド 192・重なり 64 px をブレンド。各タイル内で因果ループを回し、タイルごとに cache をクリア（空間タイル × 時間チャンクの 2 軸分割） | `autoencoder_kl_wan.py`（v0.40.0）`:1076-1081`・`1331-1395`                            |
| 軽量デコーダ           | LightVAE / LightTAE / taew2_1                                                                                                                                                      | <https://huggingface.co/lightx2v/Autoencoders>、<https://github.com/madebyollin/taehv> |

- H100・bf16・81 フレームの実測（lightx2v）: Wan2.1 VAE はデコード 5.46 秒 / 10.13 GB・エンコード
  4.17 秒 / 8.50 GB、lightvaew2_1（75% 枝刈り + 蒸留）はデコード 2.07 秒 / 5.57 GB、lighttaew2_1 と
  taew2_1 はデコード 0.25 秒 / 0.41 GB。TAEHV は品質がやや劣ることを明記する。TAEHV README の比較
  （約 6〜9 GB → 0.5 GB 未満）は Hunyuan VAE・GH200 の値で Wan の値ではない。
- 「Wan の VAE はタイル非対応」という検索要約は古い情報。
- 論文は Wan-VAE を「HunyuanVideo の VAE より 2.5 倍速い」とする（arXiv 2503.20314 §4.1・WebFetch 要約）。

## 8. MiniMax H3

### 8.1 用語

- **single-stream DiT**: テキスト・条件・映像・音声の潜在を 1 本のトークン列にまとめて self-attention
  する Transformer。Wan のような cross-attention は無い。
- **CFG 蒸留**（guidance-distilled）: CFG の効果を学習で取り込み、1 ステップ 1 forward で済むように
  したもの。negative prompt を使わない。
- **SwiGLU**: ゲート付き FFN の一種（SiLU のゲート × 線形）。
- **MM-RoPE**: マルチモーダル用に軸を割り当てた RoPE（詳細は未調査）。

### 8.2 正式名と発表日

- 正式名は「MiniMax H3」、API の model ID は `MiniMax-H3`。テキスト・画像・映像・音声を文脈に受け、
  ネイティブのステレオ音声つき映像を最大 2K・15 秒で出す omni-modal 生成モデル
  （<https://huggingface.co/MiniMaxAI/MiniMax-H3> README L42・L57、
  <https://platform.minimax.io/docs/api-reference/video-generation-v2-create>）。
- 日付（一次）:
  - 2026-07-31 発表（公式ブログ <https://www.minimax.io/blog/minimax-h3>。この時点では「We plan to
    open up the model weights in the coming days」）。
  - 2026-08-02 ライセンス日付（LICENSE L2「release date/License date: August 2, 2026」・L63「Last
    revised」）。
  - 2026-08-03 open-source 化の公式 news（<https://www.minimax.io/news/minimax-h3-open-source>・
    WebFetch 要約。配布先は HF と ModelScope）。
  - HF リポの createdAt は 2026-07-28（非公開期間を含むと推測）、lastModified 2026-08-13、sha 42ed227e
    （HF API）。GitHub `MiniMax-AI/MiniMax-H3` は 2026-07-30 作成のミラー。
- Hailuo との関係: MiniMax のブログは Hailuo-02 のアーキテクチャを捨てて作り直したと述べる（「set
  aside the Hailuo-02 architecture」）。Hailuo-02 / 2.3 / T2V-01 / T2V-01-Director は v1 API の model ID
  として残り、H3 は v2 エンドポイント（`/v2/video_generation`）
  （<https://platform.minimax.io/docs/api-reference/video-generation-t2v>）。

### 8.3 open-weight の範囲とライセンス

- HF `MiniMaxAI/MiniMax-H3` は gated=false・280 ファイル（HF API）。MiniMax の動画モデルで open-weight
  なのは H3 だけ（HF の org 21 件・GitHub org を観測）。
- 公開物は H3-Base の 2 チェックポイント（README L80-82・L114・L148・L160・L168・L178-179・L282-284）:
  - **FL2VA**: text / 先頭・末尾フレーム → 映像 + 音声（t2va / fl2va）。
  - **Ref2VA**: オムニ参照（ref2va）。
  - どちらも出力 768p。
- 未公開（API でのみ提供）:
  - **H3-Context-IR**: 多段のホスト型前処理。README は品質に critical と明記。
    `/video-generation-v2-h3-context-ir`。
  - **H3-Regenerate-2K**: 2K 再生成（「not yet open-sourced」）。`/video-generation-v2-regeneration`。
    ローカルの H3-Base と API を組み合わせる Full 2K Workflow が README L260-284 にある。
  - **sparse attention**: 「will be released in a future update」。初回公開は full attention のみ。
- 形式は同じ重みの 2 形式（README L200-206、`model_index.json`）:
  - 原形式: `FL2VA/`・`Ref2VA/`（SGLang / vLLM 向け・`MiniMaxH3DiTModel`・diffusers 0.32.2）。
  - diffusers 形式: ルート直下の `transformer/`（t2va・fl2va）・`transformer_ref/`（ref2va）・
    `text_encoder/`・`vae/`・`audio_vae/`・`tokenizer/`・`processor/`・`scheduler/`・
    `audio_scheduler/`。`MiniMaxH3ModularPipeline`・`_diffusers_version 0.36.0.dev0`。diffusers の main
    には Modular Diffusers 版のみ統合済み（DiffusionPipeline 版は無い・
    `src/diffusers/models/transformers/transformer_minimax_h3.py` 666 行、
    <https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3>）。
- **ライセンス**（MiniMax H3 Community License Agreement・2026-08-02・許諾者 Nanonoble Pte. Ltd.・
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE>）:
  - Applicable Territory = 全世界から Excluded Territories を除いた地域（L8）。Excluded Territories は
    EU・英国・韓国・米国（L10）。日本への言及は無く、除外地域に含まれない。
  - 年商 2,000 万 USD 超の商用利用は事前の書面許可が要る（L36）。
  - 商用 UI には「MiniMax H3」の表示が要る（L37）。配布物には「Powered by MiniMax H3」の表示が要る
    （L29）。
  - 出力を他の AI モデルの改善に使うことを禁止。Model Derivatives には蒸留や合成データ学習も含む。
  - 再配布には NOTICE の同梱と改変ファイルへの明示が要る。利用者を同等以上の制限で拘束する義務
    （V.2）、第三者に生成させるサービスには safeguards の実装（V.5）。
  - 再配布は「solely within the Applicable Territory ... to Third Parties within the Applicable
    Territory」に限る（III 節冒頭・L25）。許諾地域の外での使用・配布・出力の表示も許諾外（L42・観測:
    scratchpad の LICENSE 写し）。
  - 地域除外は暫定（`docs/QA-about-License.md` L20「The current limitation means "not yet", not "not
    ever."」・L40）。除外地域向けの個別ライセンスの問い合わせ窓口がある（LICENSE L23）。
  - （推測・法的助言ではない）日本でのローカル推論・改変（量子化など）は条文上可能に読める。ただし
    量子化済み重みを HF のような全世界向けの場で再配布すると、ジオフェンスできない公開配布として III
    節に抵触しうる。ブラウザで第三者に生成させるデモ（WebGPU で重みを DL させる形）にも V.2 / V.5 と
    L42 が掛かる。法務確認か MiniMax への問い合わせ（model@minimax.io）が要る。

### 8.4 構成とサイズ

| 部品                            | 構成                                                                                                                                                                                                                                                                                        | サイズ                                         | 根拠                                                                                                                        |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| H3-Omni-Transformer（本体 DiT） | 33B dense single-stream、うち約 13B が AdaLN 関連の枝。hidden 5376・50 層・token refiner 2 層・56 heads × 128・SwiGLU 14336（bias なし）・RMSNorm + qk-norm・3D MM-RoPE・patch [1,2,2]・in_channels 24・audio_in_channels 32・text_dim 5120・rope_freq_dim 16・rope_theta 1e4・CFG 蒸留済み | BF16 66.28 GB（GiB で 61.7）                   | README L142-146・L181、`transformer/config.json`、`transformer_minimax_h3.py` L36・L103-131・L233-234・L273・L319-371・L443 |
| H3-Encoder（テキスト）          | Qwen3-VL-32B を丸ごと使い、正規化前の 50 層目の hidden state（5120 次元）を読む。テキスト部 64 層・GQA 64/8、ビジョン部 27 層。tokenizer に `<d>` などの特殊トークンを追加                                                                                                                  | BF16 66.71 GB（66,714,780,128 B・GiB で 62.1） | README L118-120、`text_encoder/config.json`、diffusers `modular_pipelines/minimax_h3`                                       |
| H3-VisualVAE（映像）            | 時間因果の f16t4d24（空間 16×・時間 4×・24ch。patchify 込みで空間 32×）。エンコーダは CNN、デコーダは後から学習した ViT（36 層・32 heads × 64 = 2048 次元・register tokens 4）。17 フレームのクリップ単位（token_drop 3）                                                                   | 全 F32・約 2.60B params・10,415,635,127 B      | README L128-132、`vae/config.json`、観測（ヘッダ: 1,265,249,280 + 1,238,985,648 + 99,634,056 params）                       |
| H3-AudioVAE（音声）             | 32 kHz の各チャンネルを独立に 40 Hz・32ch の潜在へ圧縮しステレオに組み直す。DAC / BigVGAN 系（encoder_rates [2,4,4,5,5] の積 800）                                                                                                                                                          | 605,431,611 B（0.61 GB）                       | README L136-138、`audio_vae/config.json`                                                                                    |

- AdaLN は (timestep, modality) ごとの 6 パラメータを行ごとに `index_select` する（modality 0 = video・
  1 = text・2 = audio、`adaln_out_features` 96768 = 5376 × 18）。一部の入出力 head は F32、残りは BF16
  （観測: shard1 は F32 12 / BF16 52 テンソル）。
- AdaLN の modulation 出力は timestep だけで決まるので事前計算・キャッシュでき、推論専用なら約 13B 分
  を読み込まなくてよい（README L142「these parameters do not need to be loaded for inference-only
  deployment」）。推論に要るのは実質約 20B。キャッシュの作り方と容量は未確認。
- （推測）テキストエンコーダは `hidden_states[50]` しか読まないので、51〜64 層と lm_head は計算上不要。
- **出力仕様**: 24 fps・4〜15 秒（diffusers は 5〜15 秒・§8.5 補正 4）。短辺は既定 768（学習時の
  canvas 1344×768）・縦横は 32 の倍数。フレーム数は 17n+5 に切り上げ（既定 124 = 17×7+5 → 潜在
  37 フレーム・潜在のフレーム数は 5n+2）。scheduler は flow 系で shift は video 12.0 / audio 3.0。
  SGLang の既定は 50 steps（README L61-68、scheduler config、
  <https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3>）。
  - 導出: 1344×768・124 フレームで映像トークンは 42 × 24 × 37 = 37,296 行、音声は約 207 潜在 × 2ch。
  - 実測例: RTX 5090 で 864×480・124 フレーム・20 steps に 112.2 s（SGLang cookbook）。diffusers docs は
    「960x544 runs about 2.3x faster per step than the trained 1344x768」。
- **必要メモリ**（diffusers docs の Memory 節・README L227-247）: BF16 では transformer 61.7 GiB と
  Qwen3-VL 62.1 GiB で、80 GB 1 枚に同時には載らず auto CPU offload が前提。24〜32 GB のカードでは
  torchao の int8 weight-only + block 単位の group offload で、ホスト RAM 約 75 GB が要る。SGLang の
  README 例は 4 GPU（Ulysses 並列）。FP8 は B200 / B300 でのみ検証済み。
- **リポ容量**: tree API の全ファイル合計は 498,474,749,480 B（約 498 GB・観測）。これは原形式 2 本
  （FL2VA / Ref2VA それぞれに text_encoder 66.7・transformer 66.3・video_vae 10.4・audio_vae 0.6 GB）と
  diffusers 形式（transformer と transformer_ref 各 66.3 GB、text_encoder・VAE も重複）の 3 重保持込み。
  diffusers で t2va だけを読む実ダウンロードは 66.28 + 66.73 + 10.42 + 0.61 ≈ **144 GB**（観測）。
- design.minimax.io/h3 にある「pruned int8 checkpoint (~42 GB)」の配布先は未確認（HF の MiniMaxAI org
  には無い）。
  - 2026-09-10 の事前調査にも「42.5 GB」が出るが、それは同調査が「公式配布物のサイズとする根拠には
    ならない」と退けた従来の索引の値（`docs/research/2026-09-10-codex-mtp-optimization.md:447`）。
    この 42.5 GB と pruned int8 の ~42 GB が同じものを指すかは、根拠が無く分からない。

### 8.5 敵対的裏取りの結果と 4 つの精度補正

深掘りレッグが掃引レッグの主張を反証するつもりで一次ソースに当たった。存在・公開状況・ライセンスの
除外地域・アーキテクチャ・VAE・音声 VAE・open-weight が H3 だけである点は、すべて holds。refute には
至らないが、精度を上げる補正が 4 つあった。

1. **H3-Max**（refuted・部分的・severity 低）: fal が H3 の open-weight から post-train した版で、重みが
   未公開なのは holds（<https://blog.fal.ai/introducing-h3-max-by-fal/>・2026-08-27）。ただし「fal の
   API のみ」は不正確で、MiniMax 公式の v2 API も `MiniMax-H3-Max` を受け付ける（480P / 768P・5〜15
   秒・2K 非対応）。
2. **「BF16・diffusers 形式」**（holds・補足付き・severity 低）: 原形式（SGLang / vLLM 用）も同梱し、
   映像 VAE は F32。HF API の safetensors 要約（BF16 33.1B・F32 17.2M）は一部のファイルしか数えて
   おらず、全体の dtype 構成の根拠には使えない。
3. **リポ 498 GB**（holds・補足付き・severity 中）: 重複込みの値で、1 タスクの実ダウンロードは約
   144 GB（§8.4）。WebFetch の要約モデルは合計を「210-220 GB」と誤って返した（jq の機械集計が正）。
4. **尺の下限**（uncertain・severity 低）: README と API は 4〜15 秒、diffusers は 5〜15 秒。ローカルで
   4 秒（96 フレーム → 17n+5 切り上げで 107 フレーム ≈ 4.46 秒）が通るかは未検証。（推測）diffusers が
   17n+5 の格子との兼ね合いで下限を 5 秒にしている可能性。

### 8.6 「H3」の表記ゆれ

- 「Hailuo 3.0 / Hailuo 03 / Hailuo 3」は第三者の呼び名で、モデル自体は同一（二次:
  <https://runway.com/product/models/minimax-h3>「MiniMax H3 (Hailuo 3)」、
  <https://huggingface.co/blog/ResterChed/minimax-h3-hailuo-3-0>「one model, two names」）。MiniMax の
  一次資料（HF / GitHub README・ブログ・news・design.minimax.io/h3）に Hailuo 3 系の表記は無く、
  Hailuo はアプリのブランド（hailuoai.video）としてだけ出る（README L29・L53）。
- 同名衝突の候補: 同社の LLM「MiniMax-M3」（2026-06-02・image-text-to-text）と音楽生成
  「MiniMax-Music3」（2026-08-07・text-to-audio）は別物として実在（HF org 一覧）。ML 一般で「H3」と
  呼ばれる SSM 言語モデル（Hungry Hungry Hippos・2022）は記憶ベースで今回取得していない。
- 文脈（MiniMax・動画）と一意な repo 名から、「MiniMax の H3」= 動画モデル MiniMax H3 と判断した。
- MiniMax 自身の H3 技術報告は arXiv で見つからない（ヒットは H3 を評価対象にした第三者の論文
  arXiv 2609.18323 など・本文未精読）。ブログは「近日共有」と予告していた。仕様の一次の正本は HF の
  README・config・diffusers 実装と扱う。

## 9. 未確認・取得失敗

### 9.1 未確認（Wan2.1 の数値と挙動）

- shift / guide をどの値で参照出力を固定するか（公式既定 5.0 / README 推奨 8〜12 と guide 6 /
  Diffusers config 3.0）。
- 公式の float64 RoPE・時刻埋め込みと fp32 固定区間を f32 / f16 に落としたときの誤差。f16 で DiT の
  残差流が値域に収まるか。参照の精度を f32 と bf16 のどちらに置くか。
- 公式 VAE（float32）を f16 化したときの再構成品質。
- torch.export で diffusers の `WanTransformer3DModel` を切ったとき、RoPE の strided 代入
  （`slice_scatter` 系か）・`torch.nn.RMSNorm`・SDPA・FP32LayerNorm がどの aten op に分解されるか。
- 初期ノイズの再現性（CUDA の Generator）。参照出力との照合用のノイズ注入経路。
- T5 を有効長だけで回した結果が 512 で回した結果と一致するか（導出のみ）。
- HF transformers の `UMT5EncoderModel` の実装（スケーリング無し・層ごとの相対バイアス・
  `scalable_attention=true` の意味）は取得しておらず、公式 t5.py と config から推定した。
- umT5-XXL を f16 で回したときに overflow / NaN が出るか（報告が食い違う）。
- 1.3B DiT の weight-only 量子化（int8 / int4）の品質の数値評価。
- Self Forcing などの蒸留重みのライセンスと HF 上の配布形式。
- `google/umt5-xxl` トークナイザのライセンス表記。
- M2 無印・RTX 機での Wan2.1 1.3B の実測（s/step・総所要）。
- Chrome が利用者の M2 / RTX 機で実際に返すアダプタ上限。

### 9.2 未確認（karume 側）

- Wan の adaLN を export したノード列が融合ルール adaln の固定順と一致するか。
- Wan の interleave 形 RoPE を half-split へ寄せる手（q/k 射影の出力チャネルの並べ替え）が eager 同値で
  成り立つか。不可なら irodori 型の実数化で非融合実行になる。
- Anima の VAE（`AutoencoderKLQwenImage`）と Wan2.1 VAE の構造の同一性（リポ内に記述が無い・推測で
  同系統と見るが未検証）。
- feat_cache の持ち越し方式（グラフ入出力でホスト往復か、state スロット〈ADR 0066 / 0067・rank ≤ 4〉
  か、新しい機構か）。
- conv3d の入場門（拡張分子層 + ADR か、`aten.convolution` の attr 変種として Core ATen 層か）。
- VAE の時間方向を conv3d の n 軸（Tout·Hout·Wout）に畳むか z 軸（バッチ）に置くか（dispatch 上限と
  B タイルの再利用のトレードオフ・実測が要る）。
- conv3d の implicit GEMM で geometry profile の conv2d の欄を共有してよいか。
- umT5 の head ごとの相対位置バイアスを分解経路で通したとき、row-block 融合に乗るか、const 32 MiB
  上限に当たるか（export 実測が要る）。
- umT5-XXL の export に要るホスト RAM（E4B 通常版で 48 GB 以上の前例・未実測）。
- S = 32,760 の DiT 1 forward の GPU 時間と、単発 dispatch（FFN 0.9 TFLOP 級）の TDR 余裕。
- FFN 中間（f16 560 MiB）・VAE の nearest ×2 出力（f16 585 MiB）・feat_cache 合計（f16 1.8 GiB）を
  どこまで行方向やタイルで分割するか。
- cross-attention の K/V はステップを跨いで不変だが、runtime 側に step 間で再利用する経路は無い
  （self-attention の K/V はステップごとに変わる — 2026-09-10 の事前調査）。

### 9.3 未確認（MiniMax H3）

- H3 Technical Report の公開有無（ブログで「近日共有」と予告）。sparse attention の方式・AdaLN
  キャッシュの具体手順。
- HF への重み公開日時（一次は LICENSE の 2026-08-02 だけ。8/3 は公式 news と二次ソース）。
- ライセンス III 節の再配布条項が、HF への量子化重みの公開やブラウザデモでの重み配布にどう適用されるか
  （法務確認が要る）。
- AdaLN キャッシュの粒度（step 数 × modality × 6 × 5376 × 50 層）とそのサイズ、推論に要る 20B の内訳。
- design.minimax.io/h3 の「pruned int8 checkpoint (~42 GB)」の配布先。
- ModelScope 側の配布物が HF と同一か。
- 尺の下限 4 秒がローカル推論で通るか。

### 9.4 取得失敗・要約の誤り

| 対象                                                                                          | 状態                                                                               |
| --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| <https://api.github.com/search/code?q=MiniMaxH3Transformer3DModel+repo:huggingface/diffusers> | 401 Requires authentication（contents API で代替取得済み）                         |
| `https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/<sha>/scripts/readme`                    | Entry not found（ディレクトリのため。個別の .sh は未取得）                         |
| <https://x.com/i/trending/2083947871305715931>                                                | 未取得（検索結果のみ）                                                             |
| <https://www.instasd.com/post/wan2-1-performance-testing-across-gpus>                         | HTTP 403 Forbidden                                                                 |
| <https://www.w3.org/TR/webgpu/> の WebFetch 要約                                              | maxBufferSize の既定を 2^31 と誤報。curl で取得した本文で 256 MiB に訂正           |
| Wan2.1 README の計算効率表                                                                    | 画像のため本文から取得できず、HF の `assets/comp_effic.png` を画像として読んで代替 |
| H3 リポ合計の WebFetch 要約                                                                   | 「210-220 GB」と誤報。tree API の機械集計（498 GB）が正                            |

- WebFetch の要約経由で、引用の逐語性が中程度に留まるもの: Self Forcing 論文（arXiv 2506.08009）・
  QuantSparse（arXiv 2509.23681）の表・Wan 技術レポート（arXiv 2503.20314）・MiniMax の公式 news・
  v2 API ドキュメント。
- 検索スニペットのみ: `huiwon/droid_wan_umt5_cache`・<https://wan27.org/blog/wan-2-2-gguf-guide>・
  <https://runaihome.com/blog/wan-video-local-ai-gpu-guide-2026/>・Chrome（M3 Pro）の上限値の二次記事。

## 10. 裁定が要る点

裁定はしない。論点と選択肢だけを並べる。

1. **最終目標 H3 の扱い**
   - a. 構造調査に留める（2026-09-10 の事前調査と backlog の現行記述のまま）。
   - b. H3 の部品のうち単体で意味のあるもの（映像 VAE の ViT デコーダなど）だけを後の候補に置く。
   - c. 民生機でのローカル実行（ネイティブ・offload 前提）まで含めて目標に残す。
   - 判断材料: 1 タスク約 144 GB・本体 33B（AdaLN 除き約 20B）・Qwen3-VL-32B・再配布条項（§8.3・§8.4）。
2. **conv3d の追加方式**
   - a. 案 A: IR に op `conv3d` を足す（拡張分子層 + ADR の公算・src 約 1,000 行規模の推測）。
   - b. 案 B: recipe で conv2d に分解する（新 op なし・中間 kt 倍）。
   - c. 案 A の入場門を `aten.convolution` の attr 変種として Core ATen 層に置く読み（前例なし）。
   - 付随: feat_cache の持ち越し方式（グラフ入出力 / state スロット / 新機構）。深掘りレッグの提案は
     グラフ入出力の形で、潜在 1 フレームと cache テンソル群を入力にし出力フレームと更新後の cache を
     返す 1 チャンク分のグラフを、最初のチャンク用とそれ以降用の 2 種類 export する（§5.3）。
   - 付随: upsample3d の時間インターリーブ（rank 6）の rank 4 以下への書き直し（§4.3）。
3. **テキストエンコーダの方式**（§6）
   - a. GPU で i8（per-channel）に量子化。
   - b. GPU で i4（group）に量子化。
   - c. GPU で f16 のまま載せる（約 11.4 GB）。
   - d. 事前計算した埋め込みを配る（固定プロンプト）。
   - e. CPU で回す。
   - 付随: 活性を f32 にするか f16 + クランプにするか。
4. **蒸留版か原版 50 ステップか**
   - a. 原版（50 ステップ × CFG 2 回 ≈ 28.3 PFLOP）。参照実装と数値照合がしやすい。
   - b. 蒸留版（FastWan・Self-Forcing など 4 ステップ級）。ノイズ除去が律速でなくなり、T5 と VAE が
     支配的になる。重みのライセンスと配布形は未確認。
5. **最初の到達目標**
   - 解像度（832×480 か、より小さい形か）。
   - フレーム数（81 か、33 / 9 など 4n+1 の小さい値か）。
   - 対象機（RTX 機 2 GiB−4 束縛 / M2 4 GiB−4 束縛 / WebGPU 既定 128 MiB を前提にするか）と実行環境
     （Deno か Chrome か・TDR / ウォッチドッグの扱い）。

## 11. 参照

### 11.1 Wan2.1 / Wan2.2

- <https://github.com/Wan-Video/Wan2.1>（main: `wan/modules/model.py`・`vae.py`・`t5.py`・`attention.py`・
  `tokenizers.py`、`wan/configs/`、`wan/text2video.py`、`wan/utils/fm_solvers_unipc.py`、`generate.py`、
  README）
- <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B>（sha 37ec5126・`config.json`・`assets/comp_effic.png`）
- <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>（`model_index.json`・`transformer/`・
  `vae/`・`text_encoder/`・`scheduler/scheduler_config.json`）
- diffusers main: `src/diffusers/models/transformers/transformer_wan.py`・
  `models/autoencoders/autoencoder_kl_wan.py`・`pipelines/wan/pipeline_wan.py`・`models/embeddings.py`・
  `schedulers/scheduling_unipc_multistep.py`、v0.40.0 の `autoencoder_kl_wan.py`、
  <https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan>
- <https://arxiv.org/abs/2503.20314>（Wan 技術レポート）
- <https://github.com/Wan-Video/Wan2.2>・<https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>
- <https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/functional.py>（`F.normalize`）

### 11.2 先行事例・相場

- <https://github.com/mlc-ai/web-stable-diffusion>
- <https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/wan.md>・
  <https://github.com/leejet/stable-diffusion.cpp/discussions/1000>・
  <https://github.com/leejet/stable-diffusion.cpp/discussions/868>
- <https://github.com/Blaizzy/mlx-video>・<https://github.com/bhubbard/mlx-video-rs>・
  <https://note.com/mikai_daichi/n/nab2a5d452f83>
- <https://kennycason.com/posts/2025-05-20-wan2.1-on-macos.html>
- <https://wiki.drawthings.ai/wiki/Video_Generation_Basics>・
  <https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural>
- <https://huggingface.co/samuelchristlie/Wan2.1-T2V-1.3B-GGUF>・
  <https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged>・
  <https://huggingface.co/Kijai/WanVideo_comfy>・<https://huggingface.co/city96/umt5-xxl-encoder-gguf>
- <https://arxiv.org/html/2509.23681>（QuantSparse）・<https://arxiv.org/html/2503.19897v1>
- <https://github.com/sb1992/mlx-umt5>
- <https://github.com/huggingface/transformers/issues/5651>・
  <https://github.com/huggingface/transformers/pull/10956>・
  <https://github.com/kijai/ComfyUI-WanVideoWrapper/pull/1388>
- <https://raw.githubusercontent.com/ali-vilab/TeaCache/main/TeaCache4Wan2.1/README.md>
- <https://github.com/Wan-Video/Wan2.1/issues/555>
- <https://haoailab.com/blogs/fastvideo_post_training/>・<https://self-forcing.github.io/>・
  <https://github.com/guandeh17/Self-Forcing>・<https://arxiv.org/html/2506.08009v1>
- <https://huggingface.co/lightx2v/Autoencoders>・<https://github.com/madebyollin/taehv>
- <https://www.w3.org/TR/webgpu/>・<https://www.w3.org/TR/WGSL/>
- <https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys>・
  <https://toji.dev/webgpu-best-practices/device-loss.html>
- <https://onnxruntime.ai/docs/tutorials/web/large-models.html>

### 11.3 MiniMax H3

- <https://huggingface.co/MiniMaxAI/MiniMax-H3>（README・LICENSE・`docs/QA-about-License.md`・各
  `config.json`・`model_index.json`）、HF API <https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3>
  （sha 42ed227e）
- <https://github.com/MiniMax-AI/MiniMax-H3>
- <https://www.minimax.io/blog/minimax-h3>・<https://www.minimax.io/news/minimax-h3-open-source>
- <https://platform.minimax.io/docs/api-reference/video-generation-v2-create>・
  <https://platform.minimax.io/docs/api-reference/video-generation-t2v>・
  <https://platform.minimax.io/docs/guides/video-generation>
- <https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3>・diffusers main の
  `src/diffusers/models/transformers/transformer_minimax_h3.py`・`src/diffusers/modular_pipelines/minimax_h3/`
- <https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3>
- <https://blog.fal.ai/introducing-h3-max-by-fal/>
- 二次: <https://huggingface.co/blog/ResterChed/minimax-h3-hailuo-3-0>・
  <https://runway.com/product/models/minimax-h3>・<https://picsart.com/ai-models/hailuo-3/>・
  <https://comfyui-wiki.com/en/news/2026-08-03-minimax-h3-open-weights-comfyui>

### 11.4 scratchpad に落としたソース（セッション一時領域・消える前提）

`/tmp/claude-1000/-home-developer-workspace-karume/eee00bfd-9a97-4ac0-91a3-8d14df0be216/scratchpad/`
の下。

- `wan/`: 公式ソースの写し（`wan_modules_model.py`・`wan_modules_vae.py`・`wan_modules_t5.py`・
  `wan_modules_attention.py`・`wan_modules_tokenizers.py`・`wan_text2video.py`・
  `wan_utils_fm_solvers_unipc.py`・`wan_configs_*.py`・`generate.py`・`README.md`）、diffusers の写し
  （`d_tr.py`・`d_vae.py`・`d_pipe.py`・`d_unipc.py`・`d_embeddings.py`）、HF の config 類
  （`hf_*.json`）、ヘッダ集計（`dit_hdr.json`・`vae_hdr.json`）、計算スクリプト `calc.js`、
  `comp_effic.png`、Wan2.2 の `w22_readme.md`・`w22_ti2v.py`
- H3: `h3_README.md`・`h3_LICENSE`・`h3_docs_QA-about-License.md`・`h3_*config.json`・
  `h3_model_index.json`・`h3_modular_model_index.json`・`h3.json`・`tree.json`

### 11.5 karume 内

- `docs/research/2026-09-10-codex-mtp-optimization.md:409-453`・`docs/backlog.md:322-325`
- `docs/op-vocabulary.md`・ADR 0009 / 0024 / 0034 / 0055 / 0056 / 0059 / 0064 / 0065 / 0066 / 0112 / 0117
- コミット 7a725d7f・947298d0・de8bae93・cd25abf5（規模の物差し）、a71f51a9・6d0fae81（ADR 0117 段 3）
