"""Irodori-TTS v4 を torch.export 可能にするパッチ層。

対象は**テキスト条件エンコーダ**（同梱 ModernBERT-ja-310m backbone）と、Irodori 自前の
Transformer 系（speaker encoder = `ReferenceLatentEncoder` / duration predictor / DiT）。
パッチは 4 種類:

1. **ModernBertAttention の qkv 取り出しを rank ≤ 4 に落とす** — 原実装は
   `Wqkv(x).view(B, T, 3, H, D)` の **rank 5** を作ってから `unbind(dim=-3)` で 3 本に割る。
   分解後は rank 5 の `slice` になり、strided カーネルの rank 上限（`ops.STRIDED_RANK` = 4）に
   当たって export できない。最終次元を 3 等分する slice に置き換えると、同じ要素が同じ順序で
   同じ 3 本に落ちる（下の `_flat_qkv_attention_forward` の同値の根拠）。
2. **complex 形 RoPE の実数化** — 原実装（`irodori_tts.model.apply_rotary_emb`）は
   `view_as_complex` で `[B,S,H,D/2]` の complex64 を作り、complex 乗算してから
   `view_as_real` で戻す。complex は IR v1 の dtype 語彙に無く、実数対の rank 5
   （`[B,S,H,D/2,2]`）に開くと今度は rank 上限に当たる。**rank 4 のまま**「隣接ペアの入れ替え
   + cos/sin の要素倍」で書き直す（下の `_real_pair_apply_rotary_emb` の同値の根拠）。
3. **rank-2 weight の RMSNorm を rms_norm + mul へ分割** — q/k ノルムの weight は
   `[heads, head_dim]`（head ごとに別のスケール）で、IR の `rms_norm` 契約
   「weight = 正規化軸長の rank-1」（ADR 0017）に入らない。正規化そのものは最終次元
   （head_dim）1 本なので、`rms_norm`（weight 無し = ones）と weight の broadcast 乗算へ
   割る。**rank-1 weight の経路（既存の全 RMSNorm）は逐語のまま**通す。
4. **LowRankAdaLN の weightless RMS を `rms_norm` へ畳む** — `LowRankAdaLN.forward` は
   `RMSNorm` モジュールを使わず正規化を式で直書きしており（`x * rsqrt(mean(x²)+eps)`）、
   `normalize._fold_rms_norm` は weight つき rank-1 の形しか掴まないので `aten.mean.dim` /
   `aten.rsqrt.default` が IR に残る。`F.rms_norm(x, (D,), None, eps)`（weight 無し =
   ones 合成）へ差し替えると既存の畳み込み経路にそのまま乗る（下の
   `_folded_rms_low_rank_adaln_forward`）。

MUST: パッチ後のモジュールはパッチ前と **eager 同値**であること（`irodori/export.py` が
実重み・全 golden ケースで実測し、差が出たら export ごと落とす）。3 と 4 は演算列が 1 対 1 で
**ビット一致**が構造的に成り立つ。2 は式としては厳密同値だが**ビット一致は形依存**
（`_real_pair_apply_rotary_emb` の注記 — この重みの head_dim 64 では実測 0）。

MUST: パッチはクラス属性の**プロセス全域**差し替えなので、「パッチ前の参照」を採れるのは
1 プロセスにつき 1 回だけ。適用済みかどうかは {@link patches_applied} が答え、順序違反は
呼び出し側（`irodori/export.py`）が fail loudly で拒否する（恒真化 = 偽 PASS の遮断）。

NOTE: transformers のモデリングコードを差し替えるので、版は `transformers==5.14.1` に
ピンして使う（`irodori/export.py` の docstring）。
"""

from __future__ import annotations

from typing import Any

import torch

#: パッチ適用済みフラグ。プロセス全域差し替えの副作用を可視化するためだけに持つ
#: （パッチ後に「パッチ前の参照」を採ると同値検証が恒真化する）。
_APPLIED = False

#: 差し替え前の `irodori_tts.model.apply_rotary_emb`。complex 表で呼ばれた経路の委譲先で、
#: 大域を差し替えた後に元の実装を名前で引くと自分自身に戻る（無限再帰）ので捕まえておく。
_ORIGINAL_APPLY_ROTARY_EMB: Any = None


def patches_applied() -> bool:
    """このプロセスで既にパッチを当てたか。

    参照値を採る側（同値検証）が「まだ当てていない」ことを assert するための門。
    """
    return _APPLIED


def _flat_qkv_attention_forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`ModernBertAttention.forward` の同値実装（rank 5 を作らない）。

    原実装との差は **qkv の割り方 1 点だけ**で、残りは逐語（RoPE の適用・attention interface
    への引数・出力の reshape と `Wo` / `out_drop`）。

    同値の根拠: 原実装の `view(B, T, 3, H, D)` は最終次元 `3·H·D` を C 順で `(3, H, D)` に
    括り直すだけなので、`unbind(dim=-3)` の第 i 本は元の最終次元の `[i·H·D, (i+1)·H·D)` 区間
    そのもの。ここでの `slice` + `view(B, T, H, D)` は同じ区間を同じ順序で取り出す
    （コピーの有無しか違わない）。
    """
    batch, seq = hidden_states.shape[0], hidden_states.shape[1]
    heads = self.config.num_attention_heads
    width = heads * self.head_dim
    qkv = self.Wqkv(hidden_states)
    query_states, key_states, value_states = (
        qkv[..., start : start + width].view(batch, seq, heads, self.head_dim).transpose(1, 2)
        for start in (0, width, 2 * width)
    )

    from transformers.models.modernbert import modeling_modernbert as modernbert

    cos, sin = position_embeddings
    query_states, key_states = modernbert.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    attention_interface = modernbert.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modernbert.eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.head_dim**-0.5,
        sliding_window=self.sliding_window,
        deterministic=self.deterministic_flash_attn,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch, seq, width).contiguous()
    return self.out_drop(self.Wo(attn_output)), attn_weights


#: 実数形 RoPE 表の軸 1 の並び（`[S, 2, D]` の `0` = cos・`1` = sin）。表は
#: {@link real_pair_rope_table} が作り、export 台本が Tmax ぶん焼いて渡す。
ROPE_TABLE_COS, ROPE_TABLE_SIN = 0, 1


def real_pair_rope_table(head_dim: int, length: int) -> torch.Tensor:
    """`[length, 2, head_dim]` の実数形 RoPE 表（`irodori_tts` の complex 表と同じ角度）。

    上流の `precompute_freqs_cis(D, S)` が返す complex64 `[S, D/2]` の実部 / 虚部を、
    **隣接ペアへ要素複製**して `[S, D]` へ広げたもの。complex 乗算
    `(a+bi)(c+di) = (ac−bd) + (ad+bc)i` を実数の rank-4 演算で書くとき、cos / sin は
    ペアの両成分に同じ値が要る（{@link _real_pair_apply_rotary_emb}）。

    MUST: 角度の式は上流の関数から採る（写さない）— θ や次元割りが上流で変われば、写した式は
    黙って古いまま通る。
    """
    from irodori_tts.model import precompute_freqs_cis

    freqs = precompute_freqs_cis(head_dim, length)
    cos = torch.repeat_interleave(freqs.real, 2, dim=-1)
    sin = torch.repeat_interleave(freqs.imag, 2, dim=-1)
    return torch.stack([cos, sin], dim=1)


def _real_pair_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """`irodori_tts.model.apply_rotary_emb` の同値実装（complex を作らない）。

    `freqs_cis` が complex なら**原実装へそのまま委譲**する（このパッチはモジュール全域の
    差し替えなので、表を実数化していない呼び出し側の意味論を変えない）。実数表
    `[S, 2, D]`（{@link real_pair_rope_table}）で呼ばれた経路だけが下の実数式を通る。

    同値の根拠: 原実装は `x` の最終次元を隣接 2 成分ずつ複素数 `a+bi` と見なし
    `(a+bi)(c+di) = (ac−bd) + (ad+bc)i` を計算する。ここでは同じ `x` に対し

    - `x * cos` が `(a·c, b·c)`
    - 隣接ペアを入れ替えて前半を符号反転した `(−b, a)` に `sin` を掛けたものが `(−b·d, a·d)`

    となり、和は `(a·c − b·d, b·c + a·d)` で成分ごとに一致する。`x − y` を `x + (−y)` に
    置き換えても f32 の丸めは変わらない（乗算は正確丸めなので `(−b)·d == −(b·d)`）。

    NOTE: **ビット一致は形依存**。head_dim 64（この重みの全 8 層）では実測 0 だが、
    head_dim 8 では 1 ulp ずれる（`irodori/tests/test_patch.py` が両方を固定している）。
    _推測_: torch の
    complex 乗算が最終次元長ごとに別カーネルを選び、積和が FMA へ縮約される形がある —
    倍精度参照からの誤差は両者同オーダーで、どちらが正しいという差ではない。したがって
    「式として厳密同値・丸めは実測で確認」が主張の正確な形で、実重みの門
    （`irodori/export.py` の atol 0）はこの幾何での実測として立っている。

    MUST: 中間を rank 5（`[B,S,H,D/2,2]`）にしない — strided カーネルの rank 上限
    （`ops.STRIDED_RANK` = 4）に当たる。head 軸を潰した `[B,S,H·D/2,2]` で入れ替えてから
    元形へ戻す（要素順は変わらない）。
    """
    if freqs_cis.is_complex():
        if _ORIGINAL_APPLY_ROTARY_EMB is None:
            raise AssertionError("complex 表の委譲先（差し替え前の apply_rotary_emb）が無い")
        return _ORIGINAL_APPLY_ROTARY_EMB(x, freqs_cis)
    # MUST: 軸 1 の取り出しは `select` ではなく長さ 1 の `slice`。長さ 2 の軸を切る
    # `aten.select.int` は IR 語彙に無く（normalize が畳むのは長さ 1 の軸だけ）、未対応 op で
    # 落ちる。`[S,1,D]` を `unsqueeze(0)` すれば狙いの `[1,S,1,D]` に届く。
    cos = freqs_cis[:, ROPE_TABLE_COS : ROPE_TABLE_COS + 1].unsqueeze(0)
    sin = freqs_cis[:, ROPE_TABLE_SIN : ROPE_TABLE_SIN + 1].unsqueeze(0)
    pairs = x.reshape(x.shape[0], x.shape[1], -1, 2)
    swapped = torch.cat([-pairs[..., 1:2], pairs[..., 0:1]], dim=-1).reshape(x.shape)
    return x * cos + swapped * sin


def _split_weight_rms_norm_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    """`irodori_tts.model.RMSNorm.forward` の同値実装（rank ≥ 2 weight を分割する）。

    weight が rank-1 の経路は**原実装の式をそのまま**通す（`normalize._fold_rms_norm` が
    従来どおり 1 ノードへ畳む）。rank ≥ 2 の weight（q/k ノルムの `[heads, head_dim]`）だけ、
    正規化本体を `F.rms_norm`（weight 無し）へ、head ごとのスケールを broadcast 乗算へ割る。

    同値の根拠: 原実装は最終次元（head_dim）で `x·rsqrt(mean(x²)+eps)` を作ってから weight を
    掛ける。`F.rms_norm(x, (head_dim,), None, eps)` は同じ縮約・同じ式で、掛ける順序も
    変わらない（**ビット一致**が主張 — 実測 0）。
    """
    x_dtype = x.dtype
    x = x.float()
    if self.weight.ndim == 1:
        x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(x_dtype)
    normalized = torch.nn.functional.rms_norm(x, (x.shape[-1],), None, self.eps)
    return (normalized * self.weight).to(x_dtype)


def _folded_rms_low_rank_adaln_forward(
    self: Any, x: torch.Tensor, cond_embed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`irodori_tts.model.LowRankAdaLN.forward` の同値実装（正規化を `rms_norm` に寄せる）。

    原実装との差は**正規化 1 行だけ**で、残り（shift / scale / gate の低ランク補正・
    変調・`tanh` ゲート）は逐語。

    同値の根拠: 原実装の `x * rsqrt((x*x).mean(-1, keepdim=True) + eps)` は
    `F.rms_norm(x, (D,), None, eps)` の定義そのもの（weight 無し = ones）で、縮約軸も
    式も同じ。**ビット一致**が主張（`irodori/export.py` の atol 0 が実重みで実測する）。

    差し替える理由は語彙の側にある: この正規化は `RMSNorm` モジュールを経由しないので
    `normalize._fold_rms_norm`（weight つき rank-1 の形を掴む）に引っかからず、
    `aten.mean.dim` / `aten.rsqrt.default` が IR に残ってしまう。
    """
    shift, scale, gate = cond_embed.chunk(3, dim=-1)
    shift = self.shift_up(self.shift_down(torch.nn.functional.silu(shift))) + shift
    scale = self.scale_up(self.scale_down(torch.nn.functional.silu(scale))) + scale
    gate = self.gate_up(self.gate_down(torch.nn.functional.silu(gate))) + gate

    x_dtype = x.dtype
    x = x.float()
    x = torch.nn.functional.rms_norm(x, (x.shape[-1],), None, self.eps)
    x = x * (1.0 + scale) + shift
    gate = torch.tanh(gate)
    return x.to(x_dtype), gate


def apply_patches() -> None:
    """ModernBERT / Irodori 本体のパッチをプロセス全域へ当てる（冪等）。

    MUST: 呼び出し側は「パッチ前の参照」を採り終えてから呼ぶ（{@link patches_applied}）。
    """
    global _APPLIED, _ORIGINAL_APPLY_ROTARY_EMB
    if _APPLIED:
        return
    from irodori_tts import model as irodori_model
    from transformers.models.modernbert.modeling_modernbert import ModernBertAttention

    ModernBertAttention.forward = _flat_qkv_attention_forward
    _ORIGINAL_APPLY_ROTARY_EMB = irodori_model.apply_rotary_emb
    # SelfAttention.forward はモジュール大域の名前で引くので、大域を差し替えれば届く。
    irodori_model.apply_rotary_emb = _real_pair_apply_rotary_emb
    irodori_model.RMSNorm.forward = _split_weight_rms_norm_forward
    irodori_model.LowRankAdaLN.forward = _folded_rms_low_rank_adaln_forward
    _APPLIED = True
