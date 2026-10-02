"""VAE decoder の rank 4 書き直しが共有する末端の純関数（ADR 0065 決定 2・ADR 0118 検収 4）。

anima（QwenImage VAE・T = 1 の画像専用パッチ）と wan（Wan2.1 VAE・chunk グラフ）の decoder の
書き直しは、新 op を語彙に入れずに同じ 3 つの演算を表す必要がある:

- チャネル方向の L2 正規化（{@link l2_normalize} — `linalg_vector_norm` を語彙に入れない・ADR 0017）
- nearest-exact ×2 のアップサンプル（{@link nearest_exact_2x} — upsample を語彙に入れない）
- `time_conv` の出力の時間インターリーブ（{@link interleave_frames} — rank 6 を rank 4 に収める。
  今の使い手は wan だけ）

どれもモジュールの重みも family の構成も読まない純関数なので、2 つの family で写しを持つと片方
だけ直った形が作れる。正本をここ 1 本にする。

ここへ**上げないもの**: cache の仕組み・構成の門（`wan.vae_patch.assert_supported`）・上流のコードを
写した forward（RMS_norm / Resample / AttentionBlock）。family ごとに上流のクラスと前提が違う。

MUST: 書き換えは**同値**であること。データ移動だけのもの（{@link nearest_exact_2x} /
{@link interleave_frames}）は上流とビット一致、縮約の順が変わるもの（{@link l2_normalize}）は
`F.normalize` と丸めの範囲で一致する（`anima/tests/test_patch.py`・`wan/tests/test_vae_patch.py`）。

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
