# Known issues — 未解決バグ

> 置き場の規約: 未解決のバグ（意図した設計制約は [limitations.md](limitations.md)）。
> 解決したら該当節を削除し、修正コミットへのポインタを残さない（履歴は git が持つ）。

## Metal（Apple GPU）で attention i8a8 と conv2d の 2 経路一致が崩れる

実機 **Apple M2 / Deno 2.9.4** で `deno test -A packages/runtime/tests/` が 6 本赤になる
（Linux / Vulkan は全緑）。GEMM の共有タイル書き込みを静的成分へ直した後も残ったもので、
**その機序では説明できない**別の Metal 差。調査の全体は
[research/2026-08-06-metal-silent-miscompute.md](research/2026-08-06-metal-silent-miscompute.md)。

- **attention i8a8 系 4 本**（`gpu_attention_i8a8_test.ts` / `gpu_attention_pv_i8a8_test.ts`）。
  `attention-i8a8.ts` の共有配列は `array<u32>` のスカラで動的成分書き込みを持たず、しかも
  **同じ `dot4I8Packed` を使う linear i8a8（`gpu_i8a8_test.ts`）は全通過**する。それでも
  `dot4I8Packed 版とエミュ版が atol=0 で一致する` が落ちるので、整数演算に丸め差が無い以上
  QK / PV では実際に違う値が出ている。
- **conv2d parity 2 本**（`gpu_conv2d_parity_test.ts`）。implicit GEMM ↔ 直接カーネルの
  ビット一致。conv2d の B タイルは `sb[bk * 16u + bcq] = bv4` と vec4 を丸ごと書く形で、
  上記の修正対象ではない。ただし `conv2d_block` golden（atol 1e-6 / rtol 1e-5）は通るので、
  値そのものは概ね正しく**ビット一致だけが崩れている**。

実運用への影響は未確定（この状態でも Mac で正常な画像が生成できている）。ブラウザ実行は
Dawn / Tint 系で naga を通らないため、同じ症状が出るとは限らない（未検証）。

## EmbeddingGemma の batch>1 export が変換段で通らない

`karume export-embeddinggemma --batch N`（N>1）は `karume/convert.py` で fail loudly する
（B=1 は従来どおり成功）。機序は 2 段:

1. transformers（5.14 系）の `masking_utils.find_packed_sequence_indices` が、
   `Gemma3TextModel` 内部の `position_ids [1,T]` と `batch_size=N` の不一致で trace 中に
   packed-sequence 分岐へ入り、`aten.eq.Tensor` / `aten.index.Tensor` / `aten.ne.Scalar` が
   IR まで生き残る（B=1 ではこの不一致が起きず、既存の Tmax 定数 + `sym_prefix_slice`
   畳み込みに吸収される）。
2. この分岐を monkeypatch で外すと、今度は帯マスクの `aten.bitwise_or.Tensor` が
   **bool 定数を IR v1 の initializer にできない**制約（f32 / i32 のみ）に当たる。しかも
   同 patch は **B=1 でも同じエラーを誘発する** — packed-sequence 分岐の存在自体が現行の
   帯マスク定数畳み込みパターン成立の前提になっており、eager 同値のつもりの patch でも
   安全ではない（実測 A/B・2026-08-11）。

根治は convert/normalize 側の一般化（bool 定数の f32/i32 化 or initializer dtype の拡張 +
batch>1 のマスク畳み込み対応）で、コア変換基盤への設計判断が要る。`--batch` フラグ自体は
一般化が入ればそのまま使える形で維持している。

## op 追加の層分類が一部の op で自明にならない

ADR [0043](decisions/0043-op-addition-layers.md) の層定義に、実モデルの recon
（2026-08-13・BiRefNet_HR / Depth Anything V2）で **2 つの穴**が見つかった。どちらも既存 op
の扱いには影響しない（層は今後の追加手続きだけを支配する — 同 ADR 追記 決定 3）が、次に
op を足すときの判定がそのままでは割れる。

1. **Core ATen 外の原子で、要求元がモデルであるもの**（`torchvision::deform_conv2d`）が
   どの層にも入らない。第 1 層は「原子 ∩ Core ATen」、第 1' 層は「Core ATen 外**かつ**
   karume の IR / 実行モデル自体が要求するもの」（初例 `sym_prefix_slice`）と定義されており、
   モデル由来の Core ATen 外原子はその隙間に落ちる。
2. **「容量・性能で非成立」の線引きが 2 か所で非等価**。
   [op-vocabulary.md](op-vocabulary.md) の判定手順 3 は「合成が数値的に不可能 **or 容量・
   性能で非成立** = 原子（第 1 層・台帳のみ）」、ADR 0043 追記 決定 1 は「語彙内の他 op の
   合成で厳密同値に書けるなら分子（第 2 層・ADR 必須）」。`upsample_bilinear2d` は
   「matmul 2 本で厳密同値に書けるが FLOP 184 倍 / gather 形は index が 805MB〜3.22GB」で
   **両方に当たる**。既存の前例も両側にある（`leaky_relu` = 中間 1.5〜2 倍で第 2 層 /
   `batch_norm` = 537MB で原子）。

**暫定の運用（2026-08-13 ユーザー裁定）**: 第 1 層 = Core ATen 内の原子 / 第 1' 層 =
**それ以外の原子**（要求元が IR かモデルかを問わない）。`upsample_bilinear2d` は Core ATen
内なので第 1 層（台帳のみ・ADR 不要）として扱う。線引きの明文化と ADR 0043 への反映は、
リファクタリング / 整理のタイミングでまとめて再分類する。

## ここから外れたもの（記録）

- GPU の clamp / clamp_min / relu / amax / amin の NaN 非伝播 → **根治済み（2026-08-03・
  ビット列 NaN 判定）**。裁定と機序は [decisions/0020](decisions/0020-nan-propagation-bitwise.md)。
- GPUBuffer 総確保量の天井（VRAM の約 59%）→ **出所特定済み・by-design の外部制約として
  [limitations.md](limitations.md) へ移設（2026-08-03）**。Karume 側では回避不能
  （Deno がハードコードする wgpu メモリ予算しきい値）。調査記録は
  [research/2026-08-03-wgpu-memory-ceiling.md](research/2026-08-03-wgpu-memory-ceiling.md)。
