"""VAE decoder の rank 4 書き直しが共有する末端の純関数（ADR 0065 決定 2・ADR 0118 検収 4）。

anima（QwenImage VAE・T = 1 の画像専用パッチ）と wan（Wan2.1 / 2.2 VAE・chunk グラフ）の decoder の
書き直しは、新 op を語彙に入れずに同じ演算を表す必要がある:

- チャネル方向の L2 正規化（{@link l2_normalize} — `linalg_vector_norm` を語彙に入れない・ADR 0017）
- nearest-exact ×2 のアップサンプル（{@link nearest_exact_2x} — upsample を語彙に入れない）
- `time_conv` の出力の時間インターリーブ（{@link interleave_frames} — rank 6 を rank 4 に収める。
  今の使い手は wan だけ）
- Wan2.2 の残差の外側のショートカット `DupUp3D`（{@link dup_up_3d} — rank 8 を rank 4 に収める。
  今の使い手は wan だけ・ADR 0121 決定 6）

どれもモジュールの重みも family の構成も読まない純関数なので、2 つの family で写しを持つと片方
だけ直った形が作れる。正本をここ 1 本にする。

ここへ**上げないもの**: cache の仕組み・構成の門（`wan.vae_patch.assert_supported`）・上流のコードを
写した forward（RMS_norm / Resample / AttentionBlock）。family ごとに上流のクラスと前提が違う。

MUST: 書き換えは**同値**であること。データ移動だけのもの（{@link nearest_exact_2x} /
{@link interleave_frames} / {@link dup_up_3d}）は上流とビット一致、縮約の順が変わるもの
（{@link l2_normalize}）は `F.normalize` と丸めの範囲で一致する（`anima/tests/test_patch.py`・
`wan/tests/test_vae_patch.py`・`wan/tests/test_dup_up_3d.py`）。

依存方向は recipe → `_shared` の一方向だけ（ここは family を import しない）。torch だけに依存し、
diffusers は読まない（受け取るアップサンプルのモジュールは `mode` と `scale_factor` だけを見る）。
"""

from __future__ import annotations

import torch
from torch import nn

#: `F.normalize` の既定 eps（ノルムをこの値で下から抑えてから割る）。
NORM_EPS = 1e-12

#: nearest-exact アップサンプルの倍率。**整数倍のときだけ** reshape / expand と厳密一致する。
UPSAMPLE_SCALE = 2


def l2_normalize(x: torch.Tensor, dim: int, eps: float = NORM_EPS) -> torch.Tensor:
    """`F.normalize(x, dim=dim)` の同値実装（`sum` → `sqrt` → `clamp_min` → 除算）。

    `sum` が縮約軸を attrs で持つので、チャネル方向の L2 を **permute 無し**でその軸に掛ける
    （以前の anima は permute → 最終次元 sum → permute と往復しており、その 2 本が VAE decoder の
    非コアレス strided トラフィックの 99% を占めていた —
    docs/research/2026-08-04-vae-axis-reduce-recon.md §2）。

    MUST: `clamp_min(eps) → 除算` の順は原実装のまま。`+eps` に置き換えると数値意味論が変わる
    （ゼロ入力で 0 を返す性質が消える）。
    """
    norm = torch.sqrt(torch.sum(x * x, dim=dim)).clamp(min=eps).unsqueeze(dim)
    return x / norm


def nearest_exact_2x(upsample: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """nearest-exact ×2（`[B,C,H,W]`）を reshape / expand で表す（データ移動だけなのでビット一致）。

    nearest-exact は出力添字 o を `floor((o+0.5)/scale)` へ写す。scale が整数 2 なら各入力要素の
    2 連複製と厳密に一致する（非整数倍は写像が一致しないので fail loudly）。上流のアップサンプル
    （`QwenImageUpsample` / `WanUpsample`）の `x.float()` / `type_as` は f32 の活性では恒等。

    MUST: `mode` も見る。この置き換えが一致するのは nearest-exact だけで、`nearest` は出力添字を
    `floor(o/scale)` へ写す別の写像、bilinear 等は補間そのものが違う。
    """
    if upsample.mode != "nearest-exact":
        raise NotImplementedError(f"nearest-exact 以外のアップサンプルは未対応: {upsample.mode}")
    raw = upsample.scale_factor
    scale = (
        (float(raw), float(raw))
        if isinstance(raw, (int, float))
        else tuple(float(value) for value in raw)
    )
    if scale != (float(UPSAMPLE_SCALE), float(UPSAMPLE_SCALE)):
        raise NotImplementedError(f"×{UPSAMPLE_SCALE} 以外の nearest-exact は未対応: {scale}")
    batch, channels, height, width = x.shape
    wide = x.reshape(batch * channels * height, width, 1)
    wide = wide.expand(batch * channels * height, width, UPSAMPLE_SCALE)
    wide = wide.reshape(batch * channels, height, UPSAMPLE_SCALE * width)
    tall = wide.reshape(batch * channels, height, 1, UPSAMPLE_SCALE * width)
    tall = tall.expand(batch * channels, height, UPSAMPLE_SCALE, UPSAMPLE_SCALE * width)
    return tall.reshape(batch, channels, UPSAMPLE_SCALE * height, UPSAMPLE_SCALE * width)


def interleave_frames(x: torch.Tensor) -> torch.Tensor:
    """`time_conv` の 2 倍チャネルの出力 `[2C,T,H,W]` をフレーム方向へ交互に並べ直す
    → `[C,2T,H,W]`。

    `out[c, 2t+i] = x[i·C+c, t]`（上流 `reshape(b,2,c,t,h,w)` → `stack(dim=3)` → `reshape` と
    同じ並び — diffusers 0.39.0 `autoencoder_kl_wan.py:299-301`）。途中は rank 4 に収める:
    `[2,C,T,H·W]` → permute `(1,2,0,3)` → `[C,T,2,H·W]` → `[C,2T,H,W]`。データ移動だけ。
    """
    doubled, frames, height, width = x.shape
    if doubled % 2 != 0:
        raise ValueError(f"time_conv の出力チャネル {doubled} が 2 の倍数でない")
    channels = doubled // 2
    paired = x.reshape(2, channels, frames, height * width).permute(1, 2, 0, 3)
    return paired.reshape(channels, 2 * frames, height, width)


def dup_up_3d(x: torch.Tensor, *, out_channels: int, factor_t: int, factor_s: int) -> torch.Tensor:
    """`DupUp3D.forward`（チャネルの複製 → rank 8 の view / permute）の rank 4 版。

    `x` は unbatched の `[Cin, T, H, W]`、戻りは `[Cout, T·ft, H·fs, W·fs]`。並びは上流と同じで、
    `j = ((o·ft + a)·fs + b)·fs + d`・`rep = Cout·ft·fs² / Cin` と置くと
    `out[o, t·ft + a, h·fs + b, w·fs + d] = x[⌊j / rep⌋, t, h, w]`（上流 `repeat_interleave` →
    `view(B, Cout, ft, fs, fs, T, H, W)` → `permute(0, 1, 5, 2, 6, 3, 7, 4)` — diffusers 0.39.0
    `autoencoder_kl_wan.py:106-125`）。途中は rank 4 に収める: 複製 `[Cin, 1, T·H·W]` → expand
    `[Cin, rep, T·H·W]` → チャネルの添字 j の下の桁から順に W・H・T へ depth-to-space（各段
    reshape → permute → reshape）。データ移動だけなので上流とビット一致する。

    最初の chunk で時間の先頭 `ft − 1` 枚を捨てる slice（上流の `first_chunk`）はここに入れない —
    cache の仕組みに属するので wan 側（`wan.vae_patch.avg_shortcut`）が持つ。
    """
    channels, frames, height, width = x.shape
    factor = factor_t * factor_s * factor_s
    if (out_channels * factor) % channels != 0:
        raise ValueError(
            f"出力チャネル {out_channels} × 倍率 {factor} が入力チャネル {channels} で割り切れない"
        )
    repeats = out_channels * factor // channels
    volume = frames * height * width
    # 複製（上流の repeat_interleave(dim=1)）: チャネルの添字 j = c·rep + r。
    dup = x.reshape(channels, 1, volume).expand(channels, repeats, volume)
    # W の桁: j = m·fs + d → [m, T·H, W, d] → [m, T, H, W·fs]（m < Cout·ft·fs）。
    rows = out_channels * factor_t * factor_s
    y = dup.reshape(rows, factor_s, frames * height, width).permute(0, 2, 3, 1)
    y = y.reshape(rows, frames, height, width * factor_s)
    # H の桁: m = n·fs + b → [n, T·H, b, W·fs] → [n, T, H·fs, W·fs]（n < Cout·ft）。
    rows = out_channels * factor_t
    y = y.reshape(rows, factor_s, frames * height, width * factor_s).permute(0, 2, 1, 3)
    y = y.reshape(rows, frames, height * factor_s, width * factor_s)
    if factor_t == 1:
        return y
    # T の桁: n = o·ft + a → [o, T, a, H·fs·W·fs] → [o, T·ft, H·fs, W·fs]。
    plane = height * factor_s * width * factor_s
    y = y.reshape(out_channels, factor_t, frames, plane).permute(0, 2, 1, 3)
    return y.reshape(out_channels, frames * factor_t, height * factor_s, width * factor_s)
