# Known issues — 未解決バグ

> 置き場の規約: 未解決のバグ（意図した設計制約は [limitations.md](limitations.md)）。
> 解決したら該当節を削除し、修正コミットへのポインタを残さない（履歴は git が持つ）。

現在、未解決のバグは無い。

- GPU の clamp / clamp_min / relu / amax / amin の NaN 非伝播 → **根治済み（2026-08-03・
  ビット列 NaN 判定）**。裁定と機序は [decisions/0020](decisions/0020-nan-propagation-bitwise.md)。
- GPUBuffer 総確保量の天井（VRAM の約 59%）→ **出所特定済み・by-design の外部制約として
  [limitations.md](limitations.md) へ移設（2026-08-03）**。Karume 側では回避不能
  （Deno がハードコードする wgpu メモリ予算しきい値）。調査記録は
  [research/2026-08-03-wgpu-memory-ceiling.md](research/2026-08-03-wgpu-memory-ceiling.md)。
