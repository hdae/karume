# Anima チェーン recon（2026-08-02 時点）

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。

> M1-P4（Anima エクスポート + E2E）の設計材料。手法: ①プロトタイプの Anima 資産の精読
> （export 台本 / 正規化パス / パッチ層 / LoRA / テキスト資産 / エクスポート済み IR の棚卸し —
> 並列 5 レッグ）②Karume 側実コードとのギャップ分析（GPU limits・重みヘッダ・契約表の実測込み）
> ③**Karume 自身のエクスポータでの実測** — 実重み（HF `circlestone-labs/Anima-Base-v1.0-Diffusers`
> 5.3GB、キャッシュ済み）を torch.export し、既存の curated_decompositions → normalize_graph →
> 受理集合差分で未対応 aten を全件列挙。プロトタイプ側の先行 recon（実現性 recon 2026-08-01、
> 4/4 export 成功・CausalConv3d T=1 ⇒ conv2d 等価 f64 1.78e-15・f16 全重み安全を実測済み）を
> 前提とし、その主張のうち本環境で成立しないもの（§6）は訂正した。

## 1. パイプライン全容

- **4 コンポーネント**: Qwen3 text encoder（0.6B・28 層・GQA 16/8）→ AnimaTextConditioner
  （6 層 self+cross、T5 トークン列を query に [1,512,1024] を出力）→ DiT
  （CosmosTransformer3DModel 28 blocks・hidden 2048・patch(1,2,2)・AdaLN-LoRA・QK RMSNorm）
  → VAE decoder（AutoencoderKLQwenImage・空間 8 倍・CausalConv3d）。
- **グラフ外 = ホスト**: トークナイズ（Qwen2 BPE + T5 Unigram、max_length 512・テンプレート
  無し）/ FlowMatchEulerDiscreteScheduler（shift=3.0、sigmas=linspace(1,1/N,N)+終端 0）/
  timestep sin/cos 埋め込み（表計算）/ CFG（uncond+g·(cond−uncond) の 2 回逐次）/ Euler 更新 /
  latents_mean/std の per-channel 逆正規化。プロトタイプはこの「数の正」を diffusers
  v0.39 modular_pipelines から逐語で書き下すフィクスチャ生成台本（export_anima_pipeline.py）を
  持ち、**全ステップの中間 latent + 最終 image 入りの参照 safetensors** を出力する（E2E の
  段階切り分けにそのまま使える）。
- **動的次元**: text_encoder は T（min2/max512）、conditioner は Tsrc/Ttgt の独立 2 シンボル
  （取り違え検出のためケースで別々に振る）。**DiT / VAE は解像度固定の静的グラフ**
  （symbols=[]）。
- **LoRA（turbo 等）は export 前の重み焼き込み**: ΔW=(B@A)·scale を f32 で計算し元 dtype に
  in-place 加算。IR は 1 ノードも変わらずランタイム実装不要。対象は transformer と
  text_conditioner のみ。fail loudly（lora_B 全ゼロ・対象 0 件・形状不一致は例外）。

## 2. Karume 受理集合に対する実測ギャップ（一次データ）

512px・T=1・batch1・静的 shape・strict=False・パッチ無しで 4/4 export 成功。受理集合
58 target（ATEN_HANDLERS + SKIP + FOLDABLE）との差 = **未対応 19 種（和集合）**:

| コンポーネント   | ノード数（normalize 後） | 未対応 | max rank / rank≥5                           |
| ---------------- | ------------------------ | ------ | ------------------------------------------- |
| text_encoder     | 3256                     | 12 種  | 5 / 168（全て GQA repeat_kv 分解）          |
| text_conditioner | 1241                     | 8 種   | **4 / 0（唯一の無衝突）**                   |
| transformer      | 6895                     | 12 種  | 8 / 518（rank6-8 は patchify 9 ノードのみ） |
| vae_decoder      | 657                      | 8 種   | 5 / 339（T=1 軸が全域に付着）               |

未対応 19 種の内訳（消し方の群別）:

1. **SDPA safe-softmax ガード（eq.Scalar(-inf) / logical_not / any.dim / full_like — where は
   受理済み）**: 全コンポーネント。→ プロトタイプの **ガード除去パス**で消える（§4）。
2. **RMSNorm 分解（mean.dim 269 + rsqrt 269）**: pow は既存 pow2→mul が吸収。conditioner
   だけ nn.RMSNorm 由来で raw に aten.rms_norm が立つが、Qwen3 / DiT は手書き実装のため
   **preserve では畳めず FX パターンマッチが要る**（プロトタイプ結論と一致・Karume でも
   preserve 拡張の副次計測で確認）。mean.dim は**全 269 本が最終次元**。
3. **RoPE と定数生成（sin / cos / pow.Scalar / reciprocal / repeat）**: 全て入力非依存。DiT は
   葉が定数のみで FOLDABLE 拡張だけで畳める。text_encoder / conditioner は **inv_freq が
   buffer 登録**のため現行 `_classify_foldable`（葉 = lifted 定数と SymInt のみ）では畳めない
   — buffer を素の属性へ持ち上げるパッチ（プロトタイプ lift_rope_buffers）とセット。
4. **VAE 固有（convolution=conv3d 32 / conv2d 5 / linalg_vector_norm 30 / index.Tensor 3）**:
   conv3d は T=1 スライスで conv2d 化（パッチ）、チャネル L2 は permute + 最終次元 sum への
   書き換え（パッチ）、nearest-exact ×2 upsample は reshape→expand→reshape の 5 手（パッチ）
   で、**残る新カーネルは conv2d のみ**。
5. **因果マスク・padding_mask の単発（eq.Tensor / le.Tensor / ne.Scalar / index.Tensor /
   upsample_nearest2d.vec）**: 全て入力非依存 or 恒常ゼロ入力 → 定数畳み込みとパッチ
   （padding_mask はゼロ定数チャネル化・upsample はホスト追い出し）で消える。

**op 名で見えないアリティ級ギャップ（本命）**:

- **bias 無し linear が 711 本中 698 本**（text_encoder 196/196・DiT 454/454・conditioner
  48/61）。`_h_linear` はアリティ 3 固定で明示拒否 → ゼロ bias 合成（ADR 0015 の conv と
  同じ手・機構 Emitted.zero_bias は汎用に存在）で吸収。
- **affine 無し layer_norm 85 本（DiT 全数）** — weight=None/bias=None（AdaLN の scale/shift
  は別ノード）。ones/zeros 合成で arity 3 へ正規化。
- **多軸 constant_pad_nd**（VAE 32 本 rank5 3 軸 / conditioner 1 本 dim=-2）→ パッチで消える
  （VAE は conv2d の padding 引数へ、conditioner の 512 パディングはホスト側へ）。
- **rank5 の cat 1 本**（DiT padding_mask concat）→ ゼロ定数チャネル化パッチで消える。
- **片側 clamp（clamp(min=eps)）**: チャネル L2 の permute 書き換えパッチが生成。`_h_clamp`
  は両側必須で明示拒否（「需要が出たら片側 op を語彙に足す」— 需要が出た）。

i64: DiT ゼロ・VAE は添字計算のみ・text_encoder が最濃（position_ids / 因果マスク生成 —
ラッパで attention_mask を落とすと大半が入力非依存化）。ADR 0009 の境界正規化の射程内。

## 3. 収束先（プロトタイプ最終 IR の実棚卸し）

プロトタイプのエクスポート済み Anima IR（safetensors `__metadata__` 埋め込み）の op 語彙:

- text_encoder（141 ノード）: add bmm cat clone embedding expand linear mul neg permute
  **rms_norm** sigmoid slice softmax **sym_prefix_slice** view
- conditioner（625 ノード）: 上とほぼ同一 + gelu（sigmoid 無し）
- transformer（334 ノード）: + layer_norm sigmoid unsqueeze（**symbols=[] で rank≤4 完結**）
- vae_decoder（477 ノード）: add bmm clamp clone **conv2d** div expand mul permute sigmoid
  slice softmax sqrt squeeze sum unsqueeze view（rms_norm/layer_norm 無し — チャネル L2 は
  sum/sqrt/div/clamp 分解形）

Karume 46 op との差で**真の新 op は rms_norm / conv2d / clamp_min（片側 clamp）の 3 つ**
（clone/view/squeeze/unsqueeze は既存の reshape 正規化が吸収、sym_prefix_slice は実装済み）。
golden は text 3 ケース（T=5/24/48）/ conditioner 2（Tsrc,Ttgt 独立振り）/ DiT 2（t=150/699）/
VAE 2 の構成が先例。

## 4. 正規化パスの差分マップ

Karume 既存 7 パスに対し、プロトタイプから移植すべきは **6 つ**（衛生 3 種は両者同等、
SBV2 Flip 専用の flip→cat は Anima 不要）:

1. `_fold_rms_norm` — mul(weight, mul(x, rsqrt(mean(x²)+eps))) → rms_norm 1 ノード。
   **順序 MUST: `_pow2_to_mul` より前**（mul(x,x) 化でパターンが壊れる）かつ
   `_promote_scalar_operands` より前（eps の Python スカラ照合が外れる）。
2. `_drop_safe_softmax_guard` — ガードの厳密照合 + **不活性証明**（依存錐に -inf 源が無い /
   -inf が単一加算マスク由来で 2 記号長 (5,9) の実評価で全行に有限要素）ができた時だけ除去、
   できなければ fail loudly。IR は JSON 層で非有限値を拒否するためガードを op として受理する
   案は構造的に噛み合わない（folded 定数の safetensors バイナリ側は非有限可 — 要確認 §8）。
3. `_lower_unit_expand` — GQA repeat_kv の unsqueeze→expand→(clone)→view を rank≤3 expand へ。
4. `_lower_split_unbind` — RoPE unbind 形（view 分割→幅 1 slice→squeeze）を rank 下げ後の
   最終次元 slice 1 本へ。
5. `_lower_reshape_permute` — rank≥5 の reshape→permute→reshape を rank4 の隣接転置列へ分解
   （patchify/unpatchify の 9 ノードが対象）。端点 rank≥5 が残る形は NotImplementedError。
6. select.int(size1 軸, index0) → squeeze の書き換え（汎用衛生）。

rank 下げ 3 パスの発火条件 **rank≥5（=STRIDED_RANK 超）限定は安全条件**（既存グラフへの
誤爆防止線 — プロトタイプ側 docstring に MUST 明記、Karume の同値 `STRIDED_RANK=4` を再利用
し新定数は作らない）。eliminate_dead_code をパス群の途中に挟む構造差は移植時に吸収する。

## 5. パッチ層の位置づけ（patch_sbv2 との違い）

プロトタイプ patch_anima の 4 パッチ（Qwen3 マスク落とし / Conditioner マスク・512 パディングの
ホスト追い出し / DiT の timestep ホスト昇格・padding_mask ゼロ定数化・patchify rank 寄せ /
VAE の conv3d→conv2d・rank4 化・feat_cache 拒否）は **export を通すためではなく IR の質
（新 op を増やさない・rank≤4・実行時ノブをグラフに焼かない）のためのキュレーション層**。
patch_sbv2（export 可否そのものが目的）と動機が異なる点を docstring に明記する。
「パッチ後は eager 同値（--verify が実測・reference はパッチ前に採る・VAE パッチはクラス属性の
プロセス全域差し替えのため順序保護必須）」の規律は共通。

## 6. rank 方針: STRIDED_RANK=4 維持 + パス移植（引き上げ案は却下）

- rank 上限が効くのは strided 6 op のみ（permute/expand/slice/cat/sym_prefix_slice/
  masked_fill）。reshape・elementwise は rank 無制限。
- 引き上げ（4→8）は params 10→18 u32・添字分解 4→8 反復を **rank4 の値にも常時課金**し、
  WGSL スナップショット 5 本 + 決定性キー改版 + rank8 の stride 組合せ検証が要る。需要は
  patchify 9 ノード（極小テンソル）に局在。ADR 0014 が flip 負 stride 案を「blast radius
  非対称」で却下した判断と同型。
- パス移植は既存グラフを 1 ノードも変えない（rank≤4 は対象外）ため **既存 E2E が回帰網に
  なる**。落ちる形が構造的に多いと実測で判明した時のみ引き上げを再検討（片方向で安全）。

## 7. E2E 成立性（実測値）

- f32 重み: text_encoder **2.22GiB** / conditioner **0.50GiB** / transformer **7.29GiB** /
  vae_decoder（conv3d スライス後）**98.7MiB**。合計 10.11GiB。
- 実機: RTX 3080 Ti 12,288MiB。WebGPU 実測 limits: maxBufferSize 1TiB /
  **maxStorageBufferBindingSize 2GiB−4**。最大単一テンソル（Qwen3 embed 593.5MiB）も
  512px の attention 行列（64MiB）も **1024px の 1.0GiB でも上限内** —
  プロトタイプ recon の「1024px はバッファ上限超えでタイル化/解像度キャップ必須」は
  **本機・本バックエンドでは成立しない（訂正）**。制約は VRAM 総量のみ。
- **コンポーネント別セッション（逐次生成・destroy）なら f32 のまま全段成立**: DiT 単体
  512px ≈ 7.6GiB。コンポーネント間の引き継ぎは encoder_hidden_states [1,512,1024] = 2MiB
  等のホスト往復のみ。**4 コンポーネント同時常駐は 512px でもマージン ≈1.1GiB・1024px は
  不成立** — f16 格納（安全性はプロトタイプが 28.1 億要素スキャンで確定済み）は次の
  量子化マイルストーンの主題であり M1-P4 には入れない。
- ホスト RAM: 31GiB。Deno の 8GiB ArrayBuffer 確保成功を実測済み — 7.29GiB safetensors の
  一括読みにサイズ壁は無い（ロード時間と CPU/GPU 二重持ちピークは E2E で実測する）。
- tolerance は SBV2/DeBERTa の値を**流用しない**（既存 MUST）。①DiT 1 ステップ ②N ステップ
  反復 ③VAE decode 後 image の 3 段で別々に実測導出する。

## 8. 実装波への申し送り（未確定・要実測）

- DiT の QK RMSNorm に weight が実在するか（プロトタイプ最終 IR に rms_norm が立っているので
  weight ありが濃厚だが、weight 無し形が出たら ones 合成で吸収）。
- 因果マスクの additive -inf 定数が folded initializer（safetensors バイナリ）に落ちる —
  Karume の emit / goldens 経路が**バイナリ側の非有限値**を拒否しないことの確認とテスト。
- squeeze.dim / transpose.int が decomp 後に残るか（残れば reshape/permute への正規化 1 行、
  出なければ足さない）。
- 静的グラフ（symbols=[]）では FOLDABLE の 2 点評価（prefix 可換性検査）が空回りする —
  担保は --verify の eager 突合に置く（検証で畳み誤りが赤になる結線を実装波で確認）。
- conditioner の実マスク: 実パイプラインは全 1 マスクを渡す（単一プロンプト・
  padding=longest）。マスクを落とすラッパの eager 同値は --verify で実測。
- 実測スクリプト（再現用）: scratchpad の anima_probe.py（uv run --group anima、GPU 不要、
  コンポーネント別実行 — DiT は f32 で RAM ~8GiB 消費）。
