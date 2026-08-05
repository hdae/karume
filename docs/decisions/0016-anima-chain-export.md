# 0016 — Anima チェーンの export 戦略と E2E 構成

- Status: accepted（2026-08-02）
- 根拠資料: [../research/2026-08-02-anima-recon.md](../research/2026-08-02-anima-recon.md)
  （Karume 自身のエクスポータでの実測 recon — 4/4 export 成功・未対応 19 種・アリティ級
  ギャップ・VRAM/limits 実測込み）

## 決定

- **emit ターゲットは 4 本**: `text_encoder`（Qwen3、動的 T・sym_max 512）/
  `text_conditioner`（Tsrc/Ttgt の独立 2 シンボル・sym_max 512）/ `transformer`（DiT、
  解像度固定の静的グラフ — 既定 512px）/ `vae_decoder`（静的・512px）。融合ターゲットは
  作らない（コンポーネント間にホスト処理 — トークナイズ・CFG・scheduler・逆正規化 — が
  挟まるため構造的に不可能）。
- **rank 方針: STRIDED_RANK=4 を維持**し、正規化パスで rank を下げる。引き上げ（4→8）は
  rank4 の値への常時課金 + スナップショット/決定性キー改版 + 検証コストの blast radius が
  patchify 9 ノードの局所需要と非対称のため却下（ADR 0014 の flip 負 stride 却下と同型）。
  「落ちる形が構造的に多い」と実測で判明した時のみ第 2 案として再検討する。
- **正規化パスを 6 つ追加**（recon §4）: `_fold_rms_norm` / `_drop_safe_softmax_guard` /
  `_lower_unit_expand` / `_lower_split_unbind` / `_lower_reshape_permute` / select→squeeze。
  順序 MUST: rms fold は `_pow2_to_mul` と `_promote_scalar_operands` より前。rank 下げ
  3 パスの発火は rank≥5 限定（既存グラフへの誤爆防止線 — STRIDED_RANK を再利用し新定数を
  作らない）。SDPA ガードの除去は**不活性証明ができた時だけ**・できなければ fail loudly
  （ガードを op 受理する案は、IR JSON 層の非有限値拒否と噛み合わないため却下）。
- **パッチ層 `patch_anima.py` を新設**（patch_sbv2 と同じ構え・モデルパッケージ本体は
  変更しない）。目的は patch_sbv2 と異なり「export 可否」ではなく **IR の質**（新 op を
  増やさない・rank≤4・実行時ノブをグラフに焼かない）— docstring に明記する。内訳:
  ①Qwen3/Conditioner の全 1 マスク落とし（512 パディングとマスク乗算はホスト）
  ②DiT の timestep 埋め込みのグラフ入力昇格・padding_mask のゼロ定数チャネル化・
  patchify の rank 寄せ ③VAE の CausalConv3d(T=1)→conv2d 置換・rank4 化・feat_cache 拒否
  ④RoPE buffer の属性持ち上げ（lift）。**パッチ後は eager 同値（--verify が実測・
  reference はパッチ前に採る・プロセス全域差し替えの順序保護）**は ADR 0013 の規律を踏襲。
- **FOLDABLE_OPS を実測ベースで拡張**（sin / cos / reciprocal / pow.Scalar / index.Tensor /
  add.Scalar / mul.Scalar / ne.Scalar / eq.Tensor / le.Tensor を候補に、畳み frontier に
  実際に現れたものだけ足す）。**訂正（2026-08-02 波 2 実測）**: 当初の「静的グラフの担保は
  --verify の eager 突合に置く」は構造的に成立しない（--verify はパッチ前後の eager 比較で、
  畳み込みは convert() の中でしか起きない — どんな畳み誤りも映らない）。実際の担保は 3 段:
  ①記号グラフ（text 系）は `_check_prefix_commutes` の 2 点評価が本当に走る ②静的グラフ
  （DiT / VAE）は畳みが同一呼び出しの再計算で「prefix 非可換」という故障モード自体が無い
  ③静的グラフの数値の網は golden E2E（IR を GPU で実行 vs eager 期待値 — 波 3）。
- **bias 無し linear はゼロ bias 合成・affine 無し layer_norm は ones/zeros 合成**で
  アリティ 3 へ正規化（ADR 0015 の conv と同じ手筋・カーネル/契約に arity 分岐を持ち込まない。
  実測: linear 711 本中 698 本が bias 無し、DiT の layer_norm 85 本全てが affine 無し）。
- **LoRA は export 前の重み焼き込み**（`lora.py` 移植）: ΔW=(B@A)·scale を f32 で計算し
  元 dtype へ in-place 加算。IR 不変・ランタイム実装なし。fail loudly（lora_B 全ゼロ・
  対象 0 件・形状不一致は例外）。turbo LoRA の E2E は必須ゲートにしない。
- **E2E は f32 のままコンポーネント別セッション（逐次生成・destroy）**で行う（512px）。
  4 本同時常駐と 1024px は VRAM 不成立 — f16 格納は次の量子化マイルストーンの主題であり
  本フェーズに入れない（検証の切り分けを濁さない — ADR 0006 が活性 f16 を退けた理由と同じ）。
  段構成: ①コンポーネント別 golden E2E（text 3 / conditioner 2 / DiT 2 / VAE 2 ケースを
  先例に）②パイプライン参照フィクスチャ（Python 逐語台本が sigmas・timesteps_proj 表・
  各ステップ latent・最終 image を出力）に対する**少ステップ通し E2E**（ホストグルーは
  テスト内 TS）。tolerance は SBV2/DeBERTa の値を流用せず、①DiT 1 ステップ ②反復後
  ③decode 後 image の 3 段で別々に実測導出する。
- **トークナイザの TS 移植（Qwen2 BPE / T5 Unigram）は本フェーズに含めない** —
  E2E はフィクスチャの input_ids を使う。実テキスト→画像デモ（examples/anima）は
  SBV2 と同じく別タスク（資産 dump 台本のパターンはプロトタイプに実在）。

## 検討した代替案

- SDPA を preserve して融合カーネル側でガード自前実装: 新 op 1 + ガード意味論の再実装で、
  分解受理（bmm/softmax は既存）+ ガード除去パスに対し得るものが無い。却下。
- mean.dim / rsqrt を語彙に足して RMSNorm を分解のまま通す: 中間バッファ 3 本と dispatch
  5 倍が 269 箇所に乗る。rms_norm 1 カーネル + FX 畳みが一様な最短（プロトタイプ実証）。却下。
- 4 コンポーネント同時常駐の通し E2E: f32 で VRAM 不成立（実測 10.11GiB + 活性）。
  コンポーネント間の引き継ぎがホスト経由 2MiB 程度で済むため、逐次セッションで十分。却下。
