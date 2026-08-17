# 0023 — 融合 attention（段階融合・ビット同一）

- Status: accepted（2026-08-03・perf a→b→c 続行のユーザー承認下でメイン裁定）
- 対象: `attention` op 新設（契約 50 op 目）+ packages/runtime/src/kernels/attention.ts + gemm.ts 断片共有 +
  エクスポータのターゲット別 SDPA 保存
- 需要の実測: DiT 1 step の GPU 実時間（ADR 0021 診断・1024px wi8）のうち bmm 22.2% +
  softmax 2.7% + attention 由来 strided/mul ≈ 27.0% が融合対象。メモリは S と P が
  1,073.7MB × 2 同時生存（1024px・DiT / VAE 各 1 本）で denoise 中 VRAM の主成分。
  設計書 = [research/2026-08-03-attention-design.md](../research/2026-08-03-attention-design.md)。

## 決定

1. **段階融合（案 B）**: `attention(q, k, v)` 1 ノード = **3 dispatch**。
   ① QK gemm（`S = (q·scale) @ (k·scale)ᵀ` を実体化。k は `[N,D]` のまま linear 型の
   共有転置読み — 旧経路の `permute(kᵀ)` が消える）→ ② 行統計（`m = amax S` と
   `inv = 1/Σexp(S−m)` — **現行 softmax のパス①②の逐語切り出し**）→ ③ PV gemm
   （A タイル充填時に `exp(S−m)·inv` を成分ごとのスカラ式で評価 — **P を実体化しない**）。
   **flash 完全融合（案 A）は却下**: プロトタイプ実測で同モデル同解像度の step 時間が
   **+12.5%（1024px）/ +3.1%（512px）悪化**（融合 5,498 GFLOP/s vs GEMM 10,754）。
   「レジスタブロッキングを足せば GEMM 並み」という期待は、D=128 と共有メモリ 32KB の
   制約下で BR=32/BC=16 がほぼ唯一の構成（S タイル 512 要素）であることから**反証済み**
   （設計書 §4.3.1）。案 A は同一 op 契約のキー変種として後段に温存 — 入れる場合は
   **別キー + tolerance 全面再導出とセット**（ADR 0022 決定 3 の規律）。
2. **ビット同一が設計の核（MUST）**: 出力は分解経路（`mul → permute → mul → bmm →
   softmax → expand → bmm`）とビット単位で一致する。成立根拠は 4 点 — (a) ① の縮約は
   K 昇順・タイル 16（現行 bmm と同一骨格）(b) scale はタイル充填時に q/k **両方**へ乗算
   (c) ② は softmax パス①②と同一の 256 幅ツリー縮約・同一走査順・同一 identity・
   `inv = 1.0/Σ` まで同一 (d) ③ の A 要素は現行 softmax パス③と同じ式・同じ演算順。
   どれか 1 つでも崩すと丸め列が変わる。**② を online softmax 化する変更は禁止**
   （やるなら案 A と同じく別キー + 再導出）。
3. **attrs `scale` は半スケール契約**（宣言必須・既定値補完なし）: q と k の両方に掛かる
   `√scale_factor`（torch `_scaled_dot_product_attention_math` と同義）。全スケール
   （内積後 1 回）は丸め列が変わるため契約違反。エクスポータは SDPA の `scale` 引数
   （省略時 `1/√D`）から `f32(math.sqrt(scale_factor))` を計算して載せる — D=128 の
   オラクル値 `0.2973017692565918`（実 DiT IR の定数）と一致することを pytest が固定。
4. **契約の絞り**: arity 3 固定・rank-4 head-first（`q[B,H,M,D]` / `k[B,H,N,D]` /
   `v[B,H,N,D]` → `[B,H,M,D]`）・uniform f32・**D は 3 者とも同一**（torch は v だけ別
   head_dim を許すが、広げると D の取り違えが shape 検査を素通りする）・**B と H は別々に
   突合**（カーネルは B·H を 1 本のバッチ軸に畳むため、積だけ見ると取り違えが値に出ない）・
   N=0 拒否。mask / causal / dropout / GQA は**欄を作らない**（欄の不存在が「語彙に無い」を
   構造で表す）。**→ このうち「q/k/v の H 完全一致」の 1 句のみ ADR
   [0067](0067-autoregressive-attention-vocabulary.md) 決定 1 が supersede**（2026-08-17 —
   GQA / MQA は `H % Hkv == 0` かつ `H ≥ Hkv ≥ 1` の整除 broadcast で受理。`r = H / Hkv` は
   導出値のままで「欄を作らない」は維持。B 完全一致・k/v 間 Hkv 一致・D 3 者同一・
   N=0 拒否も維持）。
5. **エクスポータはターゲット別 preserved**: `export_module(…, preserved=…)` を通し、
   **Anima の transformer / vae_decoder だけ** SDPA を保存（`export_anima.TARGET_PRESERVED`）。
   既定の `PRESERVED_OP_PREFIXES` には**入れない** — グローバルに足すと text_encoder
   （−inf 折り込み因果マスク）が `_h_attention` の fail loudly で export 不能になり、
   ADR 0016 の safe-softmax ガード除去パスも死ぬ。`_h_attention` は mask / is_causal /
   dropout≠0 / enable_gqa / rank≠4 / D 不一致 / 記号 D / 非有限 scale を全件列挙で拒否
   （**→ `enable_gqa` は ADR 0067 決定 1 で受理へ反転** — 2026-08-17）。
   text_conditioner は融合可能だが GPU 時間が測定限界以下（0.10ms/18.7ms）で v1 対象外。
   SBV2 は softmax 出力が 2 回消費される構造（相対位置 value 側）で融合不能（設計書 §1.3）。
6. **波0（恒等コピー掃除）は独立させない**: SDPA 保存で decomp 由来の恒等 expand 224 本・
   恒等 permute ペア・scale mul・kᵀ permute が融合対象グラフから **IR ごと消える**。
   非融合ターゲットの残余は効果が微小（text_encoder は 1 回きり 72ms の 0.3%）で見送り。

## 検証（全て実測済み・2026-08-03）

- **ビット同一の恒久の門**: `packages/runtime/tests/gpu_attention_parity_test.ts` — 同一入力で分解経路
  （実 DiT グラフ #60〜#74 と同じノード列）と `attention` op を実 GPU で流し、
  **Uint32 ビット列の完全一致**を 5 形状（軸全異 / 全端数 / v4 タイル跨ぎ / 段別変種混成 /
  DiT 形 D=128）で assert。恒真化の門（出力が定数なら fail）付き。
- **E2E**: 全系列再 emit 後、デモ PNG sha256 が **512px/wf16・1024px/wi8 とも置換前と
  完全一致**。全 E2E 緑・**tolerance 変更ゼロ**。dispatch は 512px 1 step 3,137 → 2,633
  （−16.1% — 恒等 expand 224 / 恒等 permute ペア / kᵀ permute / scale mul / softmax /
  bmm 112 が消え、attention 3 dispatch × 56 が入った差引）。
- **golden**: `fused_attention` 新設（SDPA 保存で attention 1 ノード化・**B/H/M/N/D 全て
  異なる長さ**・最終 query 行は logit −195 の underflow 域）。既存 `attention_block` は
  **SDPA を通らない手書き分解形と判明**したため 1 バイトも動かさず、分解形のままであることを
  pytest が固定（bmm→softmax→bmm の被覆網として温存）。
- **故障注入 10 件**（注入 → 対応テスト赤 → 復元を実測）: scale 片側化 / 全スケール化 /
  amax 落とし（素朴 softmax 化）/ 行統計の workgroup 単位化 / stats 行添字ずらし /
  バッチ base 落とし / kᵀ 読み取り違え / 端タイル 0.5 埋め / grid-stride 行飛ばし /
  タイル数 1 減。全て 2 系統以上のテストが検出（詳細は設計書 §4.6 と波A 報告）。

## 検出限界（記録 — ADR 0022 の穴と同じ管理）

- **B と H の取り違えはカーネルの値に出ない**（B·H を 1 軸に畳むため）— 検出器は契約層の
  軸別突合と golden の軸 5 種類検査のみ。
- **端数 M / N は実モデル経路に無い**（DiT は全て 64 の倍数）— ユニットテストが唯一の検出器。
- **行方向 grid-stride 族の「stride を定数 1 にする」変異は縮退ハーネスで検出不能**
  （進みはするので全行を最終的に訪れる — softmax でも同じ。行を「飛ばす」変異は検出できる）。
  attention_stats 固有ではなく族全体の性質として ACTIVE_DESIGN の落とし穴に記録。
- 大きい負値 logit が行内でばらつく入力は GOLDEN_TOLERANCE（atol 1e-6）と原理的に両立しない
  （|S|≈190 の f32 内積誤差 ≈1e-5 が exp で相対誤差に転写 — 融合ではなく入力の条件数。
  分解経路でも同値で、parity テストが別途固定）。safe 化の恒常検出器は golden ではなく
  gpu_ops の大きい負値ケース（素朴形は NaN）。

## 実測効果（2026-08-03）

- **PNG sha256 門**: 再 emit 後の実 IR で 512px/wf16（`20990ae4…`）・1024px/wi8
  （`ce6c950f…`）とも置換前と完全一致 — ビット同一の主張は E2E で確定。
- 512px/wf16: DiT GPU 968 → 929ms/step（attention_qk 35 + pv 43ms が旧
  bmm+softmax+expand+mul+permute ≈ 107ms を置換）。dispatch 3,137 → 2,633/step（−16.1%）。
- 1024px/wi8: attention_qk 636 + pv 764 = 1,400ms/step が旧 ≈1,588ms を置換
  （bmm 部分は同一 gemm なので利得は softmax 1 パス化 + コピー消滅ぶん。
  **step 時間の絶対値は GPU クロック（熱）で ±20% 揺れる** — 比較は同一セッションの
  キー別比率で読むこと）。
- メモリ: 1024px の DiT / VAE 各 attention の同時生存 transient が S+P+expand の
  2,147.5MB → S 1 枚 1,073.7MB へ**半減**（P 非実体化 + 恒等 expand 消滅）。
- 再 emit: 全 8 transformer 系列 + 3 vae_decoder（重みバイト不変・IR JSON は decomp 残骸の
  消滅で −137KB・エクスポータ --verify は transformer 2 点評価ビット一致 / VAE は既知の
  2.223e-05 不変）。
- 壁時計（ADR 0024 の conv2d 置換と合算・seed 42 排他実測）: 512px 26 → **23.9s**・
  1024px 87 → **68.3s**（perf マイルストーン開始前比: 61.4 → 23.9s = 2.6× / ~290 →
  68.3s = 4.2×）。

## 参照

- 設計書: [research/2026-08-03-attention-design.md](../research/2026-08-03-attention-design.md)
  （flash 反証の定量・共有メモリ逆算・故障注入計画の全表）
- 実装: packages/runtime/src/kernels/attention.ts / packages/runtime/src/kernels/gemm.ts（断片共有）/
  packages/runtime/src/runtime/executor.ts `#encodeAttention` / tools/exporter `convert._h_attention`
- 将来: 案 A（flash）キー変種（D≤152・別キー + tolerance 再導出）/ conditioner・
  text_encoder（mask 契約拡張）への適用 / cross-attention K/V の step 跨ぎ再計算
  （エクスポータ側のグラフ分割 — 設計書 §7）

## 追記（2026-08-11）: 加算 mask 入力（決定 4 の改訂）

「mask は欄を作らない」を改訂し、**省略可能な第 4 入力 `mask`** を契約に加えた
（EmbeddingGemma の帯マスク付き双方向 attention を融合経路に乗せるため — 将来枠 :111 の
mask 契約拡張の実施）。受理は狭く:

- **f32・rank-4・shape はちょうど `[1,1,M,N]`**（加算型 `S' = S + mask`・B·H の全バッチへ
  broadcast）。bool / `[B,1,M,N]` / `[1,H,M,N]` は fail loudly — 実行時マスク
  （Irodori CFG の裁定「案 a」）を入れる波で、添字算術とセットで契約を改版する。
- 契約機構は `ContractBase.maxArity`（arity 3..4 の閉区間 — `variadic` とは別。「何本でも」
  ではなく「決まったスロットが 1 つ増えるだけ」）。
- **mask × `attentionCompute:'i8a8'` は fail loudly**（i8a8 の ①QK は epilogue を持たない —
  黙って f32 へ縮退させない）。

**ビット同一の保存**: 変更は ①QK の書き出し epilogue で `S' = fl(fl(Σqk) + mask[m·N+n])` を
1 度足すだけ。分解経路の `bmm`（S を実体化）→ `add`（mask）と**丸めの位置も回数も同一**。
②③と mask なし① の WGSL・キーは 1 バイトも動かない（スナップショット列挙で機械証明）。
mask 付き parity（5 形状・band mask・−inf 込み・故障注入 2 件）で Uint32 完全一致を門化。
ビット同一の門は **f32 経路のみ**（s16 / c16 × mask は生成・実 GPU 実行の確認までで、
恒久門は未設置 — limitations に記録）。

**エクスポータ**: `_h_attention` が上記契約の mask を受理。transformers（Gemma3）は bool の
帯マスクを渡すので、`normalize._additive_attn_mask` が `where(mask, 0, −inf)` で加算型へ
落とす — **torch 自身の bool→additive 分解と同じ op・同じ定数**で、分解経路が焼く mask
定数と initializer バイト一致（テストで固定）。保存は従来どおりターゲット別 opt-in。

**実測（EmbeddingGemma-300m・2026-08-11）**: IR 1,681 → **1,273 ノード**（bmm 48 /
softmax 24 が消え attention 24 が入る・分解残骸の reshape −144 / expand −96 ほか）。
実 dispatch 958 → **838**。GPU はほぼ中立（bmm+softmax+add ≈ 1.5ms ↔ attention 3 カーネル
1.8ms）。**wall も中立（bare 52.7 → 53.7ms・T=318 79.5 → 79.2ms）** — dispatch −120 が
壁時計に出なかったことで、「ホスト ~38ms は dispatch 数へ線形」というモデルが**反証**された
（per-run の固定費が支配 — 分解は open。research/2026-08-11-skinny-m-geometry.md §3 の
ホスト分解課題に接続）。本改訂の価値は wall ではなく、①メモリ（S+P+expand → S 1 枚）
②mask 契約の開通（ModernBERT / 実行時マスク / E2B decode の土台）③ノード数 −24% にある。
