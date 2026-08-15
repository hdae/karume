"""Anima（diffusers 版）の 4 コンポーネントを **IR の質**のためにキュレーションする層。

対象は Qwen3 text encoder / AnimaTextConditioner / CosmosDiT / VAE decoder（ADR 0016）。
差し替えは import 済みクラスの属性置換（monkeypatch）とラッパで行い、`diffusers` /
`transformers` パッケージ本体には触れない。

**`patch_sbv2` との動機の違い（ADR 0016 / recon §5）**: patch_sbv2 は「export 可否そのもの」が
目的（分岐フリー化しないと `torch.export` が data-dependent guard で落ちる）だったが、ここは
**素のままでも 4/4 export が通る**（recon §2 の実測）。それでも層を置くのは出てくる IR の質の
ためで、狙いは 3 つ:

- **新 op を増やさない** — CausalConv3d(T=1) を conv2d へ、nearest-exact ×2 を reshape/expand へ、
  チャネル L2 を最終次元 sum へ（conv3d / upsample / linalg_vector_norm を語彙に入れない）
- **rank ≤ 4 に収める** — VAE の T=1 軸を落とし、patchify の flatten を proj の前へ寄せる
- **実行時ノブをグラフに焼かない** — timestep 埋め込みをグラフ入力へ昇格し、恒常ゼロの
  padding_mask を定数チャネルへ畳む

パッチ/ラッパ一覧:

1. **Qwen3 ラッパ** — `attention_mask` を渡さない。パイプラインは単一プロンプト
   （`padding="longest"`）で全 1 マスクなので `causal & all_true == causal`、加算バイアスが
   同一テンソルになり eager はビット一致する。
2. **Conditioner ラッパ** — 全 1 マスク 2 本を渡さない（SDPA の加算バイアスが全 0 になり
   `scores + 0` はビット一致）。末尾の `min_sequence_length` パディングと出力マスク乗算は
   ホスト側（IR の pad は最終次元専用 — 語彙を広げない）。
3. **DiT ラッパ** — timestep の sin/cos 埋め込みをホスト計算のグラフ入力へ昇格し、
   `padding_mask`（パイプラインは恒常ゼロ）を latent 解像度のゼロ定数チャネルへ畳む。
   入出力は T 軸を落とした rank4。
4. **VAE decoder パッチ** — T=1 の CausalConv3d を「重みの時間方向 最終スライスによる conv2d」に
   置換して rank5 を丸ごと消す。RMS_norm / Resample / Upsample / AttentionBlock も rank4 版へ。
   `feat_cache` は T=1 では参照されないので、渡された場合は fail loudly。
5. **RoPE バッファの持ち上げ** — `inv_freq` を素の属性へ降格して定数畳み込みの葉にする。
6. **DiT トークン形ラッパ**（#21 波 T2） — patchify / unpatchify / rope 表をホストへ出した
   **追加系列**。3 の逐語ラッパは 1 行も動かさない（静的系列は既存資産のまま）。

MUST: パッチ適用後のモジュールは適用前と **eager 同値**であること（`anima/export.py --verify` が
実重みで実測する）。同値でない変更をここに置いてはならない。

MUST: VAE パッチは**クラス属性のプロセス全域差し替え**なので、「パッチ前の参照」を採れるのは
1 プロセスにつき 1 回だけ。適用済みかどうかは {@link vae_patches_applied} が答え、順序違反は
呼び出し側（`anima/export.py`）が fail loudly で拒否する（恒真化 = 偽 PASS の遮断 —
ADR 0013 の規律をそのまま踏襲）。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from karume.rope import assert_rope_lifted

#: nearest-exact アップサンプルの倍率。**整数倍のときだけ** reshape/expand と厳密一致する。
UPSAMPLE_SCALE = 2

#: VAE パッチ適用済みフラグ。プロセス全域差し替えの副作用を可視化するためだけに持つ
#: （パッチ後に「パッチ前の参照」を採ると同値検証が恒真化する — ADR 0013 / 0016）。
_VAE_APPLIED = False


def vae_patches_applied() -> bool:
    """このプロセスで既に VAE decoder パッチを当てたか。

    参照値を採る側（同値検証）が「まだ当てていない」ことを assert するための門。
    """
    return _VAE_APPLIED


# ---- ④ VAE decoder を rank4 化するパッチ -----------------------------------


def _causal_conv3d_to_conv2d(conv: nn.Conv3d) -> nn.Conv2d:
    """T=1 の CausalConv3d と等価な `nn.Conv2d` を作る（重みは時間方向の最終スライス）。

    因果パディングは時間方向の**先頭**に `2·pad_t` 個のゼロを積むので、T=1 かつ
    `2·pad_t == kt−1` なら入力の唯一のフレームが窓の末尾にちょうど来て、時間畳み込みは
    重みの最終スライスとしか掛からない。空間パディングは conv2d の padding 引数へ畳む
    （IR の pad は最終次元専用なので明示 pad をグラフに残せない — ADR 0017）。

    MUST: 等価条件は**両側**を見る。`2·pad_t < kt−1` は窓が入力に届かず（出力時間長 0）、
    `2·pad_t > kt−1` は先頭のゼロにも窓が当たって**出力が 2 フレーム以上**になる — どちらも
    「重みの最終スライス 1 枚」では表せない。時間 stride も 1 に限る（元実装が時間方向を
    間引く形は decoder に現れず、最終スライス等価の主張が別物になる）。
    MUST: 重みのスライスを**グラフではなくモジュール差し替え**で済ませる。FX で `select` を
    挟むと rank5 の元重み（3 倍の格納量）がそのまま initializer になる。
    """
    pad_w, pad_h, pad_t2 = conv._padding[0], conv._padding[2], conv._padding[4]
    kernel_t, kernel_h, kernel_w = (int(size) for size in conv.weight.shape[2:])
    if pad_t2 != kernel_t - 1:
        raise NotImplementedError(
            f"因果パディングが等価条件を外れる（2·pad_t={pad_t2}, kt={kernel_t}"
            f" — 2·pad_t == kt−1 のみ）: T=1 等価が崩れる"
        )
    if int(conv.stride[0]) != 1:
        raise NotImplementedError(
            f"時間 stride={int(conv.stride[0])} の CausalConv3d は未対応"
            "（T=1 等価は stride 1 のみ）"
        )
    # 出力の時間長がちょうど 1 であることを直接見る（上の 2 条件からの帰結の再確認）。
    time_out = (1 + pad_t2 - kernel_t) // int(conv.stride[0]) + 1
    if time_out != 1:
        raise NotImplementedError(f"T=1 入力の出力時間長が {time_out} — conv2d 等価にならない")
    flat = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        (kernel_h, kernel_w),
        stride=(conv.stride[1], conv.stride[2]),
        padding=(pad_h, pad_w),
        bias=conv.bias is not None,
    )
    flat.weight = nn.Parameter(conv.weight.detach()[:, :, -1].clone())
    if conv.bias is not None:
        flat.bias = nn.Parameter(conv.bias.detach().clone())
    return flat


def _l2_normalize_channels(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """`F.normalize(x, dim=1)` の同値実装（チャネル軸をその場で縮約する）。

    `sum` が縮約軸を attrs で持つようになったので、チャネル方向の L2 を **permute 無し**で
    書ける（`linalg_vector_norm` は語彙に入れない — ADR 0017）。以前はここで
    `permute(0,2,3,1)` → 最終次元 sum → `permute(0,3,1,2)` と往復しており、その 2 本が
    VAE decoder の非コアレス strided トラフィックの 99% を占めていた
    （docs/research/2026-08-04-vae-axis-reduce-recon.md §2）。

    MUST: `clamp_min(eps) → 除算` の順序は原実装のまま。`+eps` に置き換えると数値意味論が
    変わる（ゼロ入力で 0 を返す性質が消える）。
    """
    norm = torch.sqrt(torch.sum(x * x, dim=1)).clamp(min=eps).unsqueeze(1)
    return x / norm


def _rms_norm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`QwenImageRMS_norm` の rank4 版。gamma はパッチ時に `(1,C,1,1)` へ整形済み。

    原実装の `+ self.bias`（`bias=False` のとき Python の `0.0`）は落とす — 値を変えず
    ノード数だけ増やす加算で、符号付きゼロ以外に観測可能な差は無い。学習された bias を
    持つ構成（decoder には実在しない）は fail loudly。
    """
    if isinstance(self.bias, torch.Tensor):
        raise NotImplementedError("学習された bias 付きの RMS_norm は rank4 パッチの対象外")
    if not self.channel_first:
        raise NotImplementedError("channel_first=False の RMS_norm は rank4 パッチの対象外")
    return _l2_normalize_channels(x) * self.scale * self.gamma


def _resample_forward(self: nn.Module, x: torch.Tensor, feat_cache=None, feat_idx=None):
    """`QwenImageResample` の rank4 版（T=1 の画像専用）。

    `upsample3d` の `time_conv` 段は原実装でも `feat_cache is None` のときは実行されない
    （T=1 の初回チャンク相当）。`downsample` 系は decoder に現れない。
    """
    if feat_cache is not None:
        raise NotImplementedError("feat_cache 付きの Resample は画像専用パッチの対象外")
    if self.mode not in ("none", "upsample2d", "upsample3d"):
        raise NotImplementedError(f"decoder に現れない resample mode: {self.mode}")
    return self.resample(x)


def _upsample_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """nearest-exact ×2 を reshape / expand で表す（データ移動のみなのでビット一致）。

    nearest-exact は出力添字 o を `floor((o+0.5)/scale)` へ写す。scale が整数 2 なら各入力要素の
    2 連複製と厳密に一致する（非整数倍は写像が一致しないので fail loudly）。

    MUST: `mode` も見る。この置き換えが一致するのは nearest-exact だけで、`nearest` は
    出力添字を `floor(o/scale)` へ写す別の写像、bilinear 等は補間そのものが違う。
    """
    if self.mode != "nearest-exact":
        raise NotImplementedError(f"nearest-exact 以外のアップサンプルは未対応: {self.mode}")
    raw = self.scale_factor
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


def _attention_block_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`QwenImageAttentionBlock` の rank4 版（原実装から time 軸の出入りを外しただけ）。"""
    identity = x
    batch_size, channels, height, width = x.shape
    normalized = self.norm(x)
    qkv = self.to_qkv(normalized)
    qkv = qkv.reshape(batch_size, 1, channels * 3, -1).permute(0, 1, 3, 2).contiguous()
    query, key, value = qkv.chunk(3, dim=-1)
    attended = functional.scaled_dot_product_attention(query, key, value)
    attended = attended.squeeze(1).permute(0, 2, 1).reshape(batch_size, channels, height, width)
    return self.proj(attended) + identity


def apply_vae_decoder_patch(vae: nn.Module) -> None:
    """VAE の decode 経路を rank4・T=1 専用へ差し替える（冪等）。

    クラス属性の差し替え（**プロセス全域**）+ モジュール/パラメータの整形（インスタンス単位）。
    gamma は `(C,1,1)` / `(C,1,1,1)` 格納で、rank4 入力に対しては右詰め broadcast が崩れる
    （`images=False` は第 0 軸が C と B で衝突する）— パッチ時に `(1,C,1,1)` へ揃えて
    forward から分岐を無くす。
    """
    global _VAE_APPLIED
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageAttentionBlock,
        QwenImageCausalConv3d,
        QwenImageResample,
        QwenImageRMS_norm,
        QwenImageUpsample,
    )

    QwenImageRMS_norm.forward = _rms_norm_forward
    QwenImageResample.forward = _resample_forward
    QwenImageUpsample.forward = _upsample_forward
    QwenImageAttentionBlock.forward = _attention_block_forward
    _VAE_APPLIED = True

    # decode 経路だけを差し替える（encoder は export しないので触らない）。
    slots = [(vae, "post_quant_conv")]
    slots += [
        (parent, name)
        for parent in vae.decoder.modules()
        for name, child in parent.named_children()
        if isinstance(child, QwenImageCausalConv3d)
    ]
    for parent, name in slots:
        child = getattr(parent, name)
        if isinstance(child, QwenImageCausalConv3d):
            setattr(parent, name, _causal_conv3d_to_conv2d(child))
    for module in vae.decoder.modules():
        if not isinstance(module, QwenImageRMS_norm):
            continue
        target = (1, module.gamma.numel(), 1, 1)
        if tuple(module.gamma.shape) != target:
            module.gamma = nn.Parameter(module.gamma.detach().reshape(target))


class AnimaVaeDecoder(nn.Module):
    """画像 1 枚（T=1）の VAE decode。入出力とも T 軸を落とした rank4。

    `AutoencoderKLQwenImage._decode` の `num_frame=1` 経路と等価（`latents_mean` /
    `latents_std` の逆正規化はホスト側 — 実行時ノブをグラフに焼かない）。
    """

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.vae.post_quant_conv(latents)
        return torch.clamp(self.vae.decoder(hidden), min=-1.0, max=1.0)


def reference_vae_decode(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """**パッチ前**の diffusers 本来の decode（rank5・feat_cache 有り）での参照出力。

    `vae.decode` をそのまま呼ぶので、`--verify` は「conv3d → conv2d 等価」と「T=1 では
    feat_cache が結果に効かない」を **1 度の突合で同時に**実測する（片方だけ別経路で
    確かめると、もう片方の前提が黙って崩れる余地が残る）。
    """
    return vae.decode(latents.unsqueeze(2), return_dict=False)[0].squeeze(2)


# ---- ① Qwen3 テキストエンコーダ --------------------------------------------


class AnimaTextEncoder(nn.Module):
    """`Qwen3Model` の `last_hidden_state`。`attention_mask` は渡さない。

    パイプラインは `padding="longest"` の単一プロンプトなのでマスクは全 1 で、
    `causal & all_true == causal`。加算バイアスが同一テンソルになるため eager はビット一致
    する（`--verify text_encoder` が実測する）。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        assert_rope_lifted(model, "text_encoder")
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, use_cache=False).last_hidden_state


# ---- ② テキストコンディショナ ----------------------------------------------


class AnimaConditioner(nn.Module):
    """`AnimaTextConditioner` の本体部（マスク無し・512 パディング前）。

    全 1 マスクでは SDPA の加算バイアスが全 0 になるため、マスクを渡さない形とビット一致する。
    末尾の `min_sequence_length` パディングと出力マスク乗算は**ホスト側**（IR の pad は
    最終次元専用で、多軸 pad を語彙へ広げない — ADR 0016）。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        assert_rope_lifted(model, "text_conditioner")
        self.model = model

    def forward(
        self, source_hidden_states: torch.Tensor, target_input_ids: torch.Tensor
    ) -> torch.Tensor:
        model = self.model
        hidden_states = model.embed(target_input_ids).to(dtype=source_hidden_states.dtype)
        hidden_states = model.in_proj(hidden_states)
        position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
        source_position_ids = torch.arange(source_hidden_states.shape[1]).unsqueeze(0)
        position_embeddings = model.rotary_emb(hidden_states, position_ids)
        source_position_embeddings = model.rotary_emb(hidden_states, source_position_ids)
        for block in model.blocks:
            hidden_states = block(
                hidden_states,
                source_hidden_states,
                position_embeddings=position_embeddings,
                source_position_embeddings=source_position_embeddings,
            )
        return model.norm(model.out_proj(hidden_states))


def reference_conditioner(
    model: nn.Module, source_hidden_states: torch.Tensor, target_input_ids: torch.Tensor
) -> torch.Tensor:
    """全 1 マスクを渡した原経路の、512 パディング**前**の先頭行（参照値）。"""
    out = model(
        source_hidden_states,
        target_input_ids,
        torch.ones(1, target_input_ids.shape[1], dtype=torch.long),
        torch.ones(1, source_hidden_states.shape[1], dtype=torch.long),
    )
    return out[:, : target_input_ids.shape[1]]


# ---- ③ CosmosDiT -----------------------------------------------------------


class AnimaDit(nn.Module):
    """`CosmosTransformer3DModel` の 1 denoise step（cond / uncond のうち片側）。

    原 forward との差分は 3 点だけ:

    - timestep → sin/cos 埋め込みの計算をホストへ出し、`timesteps_proj` を入力に取る
      （実行時ノブと三角関数をグラフに焼かない）
    - `padding_mask`（パイプラインは `latents.new_zeros`）を latent 解像度のゼロ定数
      チャネルへ畳む（nearest リサイズはゼロ入力では恒等）
    - 入出力の T 軸（= 1）を落とし、patchify の flatten を proj の前へ寄せる
      （view の並べ替えのみ。rank ≤ 4 化は正規化パスが受け持つ）

    それ以外（rope / transformer_blocks / norm_out / unpatchify）は原実装のまま。
    """

    def __init__(self, model: nn.Module, latent_height: int, latent_width: int) -> None:
        super().__init__()
        if model.config.extra_pos_embed_type is not None:
            raise NotImplementedError("learnable pos embed 付きの構成は未対応（Anima は null）")
        if model.config.img_context_dim_in or model.config.use_crossattn_projection:
            raise NotImplementedError("img_context / crossattn_projection 付きの構成は未対応")
        if not model.config.concat_padding_mask:
            raise NotImplementedError("concat_padding_mask=False の構成は未対応")
        self.model = model
        # 素の属性（パラメータ/バッファにしない）— lifted tensor constant として運ぶ。
        self.padding_channel = torch.zeros(1, 1, latent_height, latent_width)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        patch_t, patch_h, patch_w = model.config.patch_size
        hidden_states = torch.cat([latents, self.padding_channel], dim=1).unsqueeze(2)
        batch, channels, frames, height, width = hidden_states.shape

        image_rotary_emb = model.rope(hidden_states, fps=None)

        patches = hidden_states.reshape(
            batch,
            channels,
            frames // patch_t,
            patch_t,
            height // patch_h,
            patch_h,
            width // patch_w,
            patch_w,
        )
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        hidden_states = model.patch_embed.proj(patches)

        temb = model.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = model.time_embed.norm(timesteps_proj)

        for block in model.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb,
                image_rotary_emb,
                None,
                None,
                None,
            )

        hidden_states = model.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = model.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (patch_h, patch_w, patch_t, -1))
        hidden_states = hidden_states.unflatten(
            1, (frames // patch_t, height // patch_h, width // patch_w)
        )
        # NOTE: 原実装のコメントどおり、この置換は patchify の逆順ではないが正しい。
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return hidden_states.squeeze(2)


class AnimaDitTokens(nn.Module):
    """`AnimaDit` の**トークン形**（解像度を持たない S 形グラフ — #21 波 T2）。

    {@link AnimaDit} との差分は入口と出口だけで、28 ブロック本体は 1 ノードも変わらない:

    - 入力が `latents [1,16,H,W]` ではなく **`tokens [1,S,68]`**（`68 = 17·2·2` —
      恒常ゼロの padding channel を足して patchify したもの。どちらもホスト側 =
      {@link dit_patchify}）
    - rope の cos / sin 表を**グラフ入力へ昇格**する（SBV2 の相対位置表と同じ手 —
      ADR 0013）。これで `model.rope` の定数畳み込みが消え、**S 依存の initializer が
      1 本も残らない**
    - 出力は unpatchify **前**のトークン `[1,S,64]`（逆並べ替えもホスト =
      {@link dit_unpatchify}）

    結果としてグラフ内に H / W が 1 つも現れず、トークン長 1 シンボル `S` の一次式だけで
    書ける（次元言語は「1 次元 1 シンボルの一次式」— docs/ir-v1.md）。

    MUST: rope 表は `[1,1,S,head_dim]` で受けて `flatten(0, 2)` で 2 次元へ戻す。
    上流の `apply_rotary_emb` は**渡された表が 2 次元であること**を前提に `cos[None,None]`
    で軸を足すので、4 次元のまま渡すと rank 6 へ膨らんで broadcast の意味が変わる。
    入力の宣言を `[1,1,S,head_dim]` にしてあるのは、静的グラフに焼かれていた定数
    （同じ shape）とホスト実装のバイト一致をそのまま突き合わせられるようにするため。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if model.config.extra_pos_embed_type is not None:
            raise NotImplementedError("learnable pos embed 付きの構成は未対応（Anima は null）")
        if model.config.img_context_dim_in or model.config.use_crossattn_projection:
            raise NotImplementedError("img_context / crossattn_projection 付きの構成は未対応")
        if not model.config.concat_padding_mask:
            raise NotImplementedError("concat_padding_mask=False の構成は未対応")
        self.model = model

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        hidden_states = model.patch_embed.proj(tokens)
        image_rotary_emb = (rope_cos.flatten(0, 2), rope_sin.flatten(0, 2))

        temb = model.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = model.time_embed.norm(timesteps_proj)

        for block in model.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb,
                image_rotary_emb,
                None,
                None,
                None,
            )

        hidden_states = model.norm_out(hidden_states, embedded_timestep, temb)
        return model.proj_out(hidden_states)


def dit_patchify(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """`latents [1,C,H,W]` → `tokens [1,S,(C+1)·pt·ph·pw]`（ホストへ出した入口）。

    MUST: {@link AnimaDit.forward} の入口（padding channel の連結 + patchify）と**逐語で
    同じ並べ替え**であること。ここが 1 軸でもずれると、S 形グラフの出力は静的グラフと
    別物になる（形は合うので shape 検査では捕まらない）。TS 側の鏡像とのバイト一致は
    `packages/runtime/tests/e2e_anima_dyn_test.ts` の「S 形 ≡ 静的グラフ」が実 GPU で押さえる。
    """
    patch_t, patch_h, patch_w = patch_size
    batch, _, height, width = latents.shape
    padding_channel = latents.new_zeros(batch, 1, height, width)
    hidden_states = torch.cat([latents, padding_channel], dim=1).unsqueeze(2)
    batch, channels, frames, height, width = hidden_states.shape
    patches = hidden_states.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return patches.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


def dit_unpatchify(
    tokens: torch.Tensor,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """`tokens [1,S,pt·ph·pw·C]` → `latents [1,C,H,W]`（ホストへ出した出口）。

    MUST: {@link AnimaDit.forward} の出口と逐語で同じ。**patchify の逆順ではない**
    （最終次元の並びが patchify 側の `(C, pt, ph, pw)` に対しこちらは
    `(ph, pw, pt, C)`）— 原実装のコメントどおりで、逆順に「直す」と静かに壊れる。
    """
    patch_t, patch_h, patch_w = patch_size
    hidden_states = tokens.unflatten(2, (patch_h, patch_w, patch_t, -1))
    hidden_states = hidden_states.unflatten(
        1, (1, latent_height // patch_h, latent_width // patch_w)
    )
    hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return hidden_states.squeeze(2)


def dit_rope_tables(
    model: nn.Module, latent_height: int, latent_width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """S 形グラフへ渡す rope の cos / sin 表 `[1,1,S,head_dim]`。

    式の正本は**上流の `CosmosRotaryPosEmbed` そのもの**（呼ぶだけで、写さない）。
    先頭 2 軸は `apply_rotary_emb` が 2 次元の表へ足す軸と同じで、静的グラフに焼かれて
    いた定数の shape と一致する。
    """
    probe = torch.zeros(1, 1, 1, latent_height, latent_width)
    cos, sin = model.rope(probe, fps=None)
    return cos[None, None], sin[None, None]


#: ホスト素表のテンソルキー（軸ごとの cos / sin）。`t` は画像（F'=1）では 1 行しか使わない。
ROPE_BASE_KEYS = ("cos_t", "sin_t", "cos_h", "sin_h", "cos_w", "sin_w")


def dit_rope_base_tables(model: nn.Module) -> dict[str, torch.Tensor]:
    """ホスト（TS）が rope 表を組むための**軸ごとの素表**を上流の式から切り出す。

    ## なぜ表を焼くのか（TS で三角関数を再計算しない理由）

    `torch.cos` / `torch.sin` の f32 実装は**正しく丸めた値とは 1 ulp ずれることがある**
    （実測: 位置 × 周波数の全 8,192 通りのうち cos 472 件 / sin 231 件が 1 ulp 差。
    例 `cos(1.0)` は torch 0x3f0a5141 に対し正しい丸めは 0x3f0a5140）。JS の `Math.cos`
    は f64 で正しく丸まるので、TS で式を写すと**必ずこの差が出る**。静的グラフには
    torch の値が焼かれている以上、S 形との**ビット同一**は表を焼く以外に成立しない
    （トークナイザを「TS で Unicode 分類を再実装せず表に焼く」のと同じ判断）。

    ## 表が解像度に依らない理由

    位置 → 角度は軸ごとに独立で、`freqs = cat([emb_t, emb_h, emb_w] * 2, dim=-1)` は
    軸ごとのブロックを並べただけ。したがって「軸 × 位置」の素表さえあれば任意の H' / W'
    の表を組める。行数は上流の `seq = arange(max(max_size))` の長さ = 位置表の**実装上の
    天井**そのもので、Anima では 128（= latent 256 = 2048px 相当）。ここを超える H' / W'
    は上流でも `repeat` の shape が合わずに落ちる。

    MUST: 切り出しは `model.rope` の**出力から**行う（式を写さない）。H'=rows / W'=1 の
    呼び出しでは w ブロックが位置 0 に固定されるので、h ブロックだけが位置で動く。
    """
    rope = model.rope
    rows = max(int(size) for size in rope.max_size)
    widths = (rope.dim_t // 2, rope.dim_h // 2, rope.dim_w // 2)
    patch_t, patch_h, patch_w = rope.patch_size

    def probe(frames: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`pe_size = [frames, height, width]` になる形で上流の rope をそのまま呼ぶ。"""
        sample = torch.zeros(1, 1, frames * patch_t, height * patch_h, width * patch_w)
        return rope(sample, fps=None)

    # 軸ごとに「その軸だけが動き、他の 2 軸は位置 0 に固定」の呼び出しを 1 本ずつ。
    cos_t, sin_t = probe(rows, 1, 1)
    cos_h, sin_h = probe(1, rows, 1)
    cos_w, sin_w = probe(1, 1, rows)

    start_h = widths[0]
    start_w = widths[0] + widths[1]
    end_w = start_w + widths[2]
    tables = {
        "cos_t": cos_t[:, : widths[0]].contiguous(),
        "sin_t": sin_t[:, : widths[0]].contiguous(),
        "cos_h": cos_h[:, start_h:start_w].contiguous(),
        "sin_h": sin_h[:, start_h:start_w].contiguous(),
        "cos_w": cos_w[:, start_w:end_w].contiguous(),
        "sin_w": sin_w[:, start_w:end_w].contiguous(),
    }
    _assert_rope_base_shapes(tables, rows, widths)
    return tables


def _assert_rope_base_shapes(
    tables: dict[str, torch.Tensor], rows: int, widths: tuple[int, int, int]
) -> None:
    """素表の切り出し位置が上流のブロック境界と合っていることを確かめる（恒真化の門）。

    切り出しが 1 ブロックずれても shape は合う（h と w のブロック幅が等しい構成では
    特に）。そこで**その軸だけが位置で動く**ことを見る: 位置 0 の行は角度 0 なので
    cos=1 / sin=0、位置 1 の行は（周波数が 0 でない限り）そこから動く。別ブロックを
    切っていれば位置に対して定数のままになり、後者で破れる。
    """
    for axis, width in zip(("t", "h", "w"), widths, strict=True):
        for kind in ("cos", "sin"):
            table = tables[f"{kind}_{axis}"]
            if tuple(table.shape) != (rows, width):
                raise AssertionError(
                    f"rope 素表 {kind}_{axis} の shape {tuple(table.shape)} が"
                    f" 期待 {(rows, width)} と違う"
                )
        if not torch.equal(tables[f"cos_{axis}"][0], torch.ones(width)):
            raise AssertionError(f"rope 素表 cos_{axis} の位置 0 が全 1 でない（切り出しずれ）")
        if not torch.equal(tables[f"sin_{axis}"][0], torch.zeros(width)):
            raise AssertionError(f"rope 素表 sin_{axis} の位置 0 が全 0 でない（切り出しずれ）")
        if torch.equal(tables[f"sin_{axis}"][1], torch.zeros(width)):
            raise AssertionError(
                f"rope 素表 sin_{axis} が位置 1 でも全 0（この軸で動いていない = 切り出しずれ）"
            )


def dit_timesteps_proj(model: nn.Module, timestep: torch.Tensor) -> torch.Tensor:
    """ホスト側で計算する timestep 埋め込み（`CosmosEmbedding` の `time_proj` 段）。"""
    return model.time_embed.time_proj(timestep).to(torch.float32)


def reference_dit(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """**パッチ前**の diffusers 経路での参照出力（`padding_mask` はパイプライン同様ゼロ）。"""
    pixels = latents.shape[-2] * 8
    return model(
        hidden_states=latents.unsqueeze(2),
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        padding_mask=latents.new_zeros(1, 1, pixels, pixels),
        return_dict=False,
    )[0].squeeze(2)
