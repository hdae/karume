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

## ここから外れたもの（記録）

- GPU の clamp / clamp_min / relu / amax / amin の NaN 非伝播 → **根治済み（2026-08-03・
  ビット列 NaN 判定）**。裁定と機序は [decisions/0020](decisions/0020-nan-propagation-bitwise.md)。
- GPUBuffer 総確保量の天井（VRAM の約 59%）→ **出所特定済み・by-design の外部制約として
  [limitations.md](limitations.md) へ移設（2026-08-03）**。Karume 側では回避不能
  （Deno がハードコードする wgpu メモリ予算しきい値）。調査記録は
  [research/2026-08-03-wgpu-memory-ceiling.md](research/2026-08-03-wgpu-memory-ceiling.md)。
