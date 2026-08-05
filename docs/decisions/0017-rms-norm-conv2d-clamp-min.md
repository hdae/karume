# 0017 — 語彙拡張: rms_norm / conv2d / clamp_min

- Status: accepted（2026-08-02）
- 根拠資料: [../research/2026-08-02-anima-recon.md](../research/2026-08-02-anima-recon.md)
- 需要の実測: RMSNorm 269 本（Qwen3 113 / conditioner 43 / DiT 113 — QK ノルム含む）、
  VAE decoder の conv2d 37 本（CausalConv3d 32 の T=1 スライス + 素の conv2d 5）、
  チャネル L2 正規化の clamp(min=eps) 30 本。プロトタイプ最終 IR の語彙差分もこの 3 op に
  収束する（recon §3）。TS 契約・Python 契約・WGSL カーネル・CPU 参照の **4 点セットを
  同一波で同時に**広げる（ADR 0015 の規律）。

## 決定

- **`rms_norm`**: arity 2（x, weight）・attrs `{eps}` 必須・正規化軸は最終次元 1 本のみ・
  weight は最終次元長の rank1・f32。`y = x · rsqrt(mean(x², 最終次元) + eps) · weight`。
  カーネルは layer_norm と同族の行 reduce（木構造縮約 — 二乗和 1 パスで足りるため
  layer_norm の「2 パス母分散」問題は生じない）。供給ルートは 2 系統を両立させる:
  ①diffusers `nn.RMSNorm` 由来の `aten.rms_norm` を PRESERVED に追加 ②手書き分解形
  （Qwen3 / DiT）は `_fold_rms_norm` パスで合成（ADR 0016）。weight 無し形が実測で出たら
  ones 合成で arity 2 へ正規化（ゼロ bias 合成と同じ手筋）。
- **`conv2d`**: 入力 [B,Cin,H,W]・重み **[Cout, Cin/groups, Kh, Kw]**・bias [Cout]
  （arity 3 固定 — bias 無しはエクスポータのゼロ bias 合成）。attrs は
  `{stride, padding, dilation, groups}` を**宣言必須・既定値補完なし**（ADR 0015 の conv1d と
  同じ規律。stride/padding/dilation は H/W の 2 成分）。出力長式は dilation を含む一般形。
  `Cin % groups == 0` / `Cout % groups == 0` / `stride >= 1` を契約検査に入れる。
  空間パディングは conv2d の padding 引数に畳まれた形しか出ない（パッチ層が保証 — IR の
  pad を多軸に広げない）。テストは **Kh≠Kw・stride/padding の H/W 非対称・Cin/Cout とも
  2 以上で互いに異なる**ケースを必ず持つ（重みレイアウト取り違えは要素数が合い shape 検査を
  素通りする — conv_transpose1d の教訓、Pitfalls 済み）。
- **`clamp_min`**: elementwise・attrs `{min}` 必須・f32。既存 clamp の attrs optional 化は
  「宣言済み attrs の既定値補完はしない + Object.hasOwn 全キー照合」（ADR 0012）を崩すため
  却下。「欠けた側を ±有限最大値で補う」は `_h_clamp` の MUST が名指しで禁じる手筋であり
  採らない。NaN の扱いは leaky_relu / clamp と同じ select 形で torch 意味論に合わせる。
  **訂正（2026-08-02 波 1 実 GPU 実測）**: 「select 形なら NaN が伝播する」は成立しない —
  シェーダコンパイラが max イディオムへ畳み、ドライバの `max` が NaN を飲む
  （`clamp_min(NaN, min=0) = 0`）。`clamp` / `relu` も同じで、`leaky_relu` が伝播するのは
  両方の枝に x が現れるためであり select 形そのものの効能ではない。乖離の実測・機序・
  根治候補（ビット列での NaN 判定）は [../known-issues.md](../known-issues.md)。
- 新カーネルはいずれも **full-write（ADR 0004 ⑤）を遵守**し、縮退ハーネス
  （packages/runtime/tests/gpu_gridstride_test.ts）へ同波で 1 本ずつ追加する（MUST — Pitfalls 索引）。

## 検討した代替案

- mean.dim + rsqrt の 2 op 追加（RMSNorm を分解のまま実行）: dispatch 5 倍 × 269 箇所と
  中間バッファ 3 本。却下（ADR 0016 と同判断）。
- conv3d カーネルの新設: CausalConv3d(T=1) ⇒ conv2d 等価がプロトタイプで f64
  max diff 1.78e-15 と実測済みで、需要が存在しない。重み格納も時間スライスで 2.8 倍縮む。却下。
- upsample_nearest 専用カーネル: reshape→expand→reshape の 5 手で厳密表現できる
  （パッチ層実装済みの先例）。新カーネル不要。却下。
