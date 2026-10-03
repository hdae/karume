"""Wan2.1 の umT5（`UMT5EncoderModel`）を有効長 `L` の 1 シンボルの IR にするラッパと、相対位置の
バケット表の Python 側の正本（ADR 0119 決定 3〜5・段 10c）。

グラフの入力は token id `[1, L]` とバケット表 `[L, L]`（{@link INPUT_NAMES}）、出力は
`[1, L, d_model]`。マスク入力は持たない（決定 4 — 有効長だけで回す）。

## 上流との差（どれも {@link Umt5EncoderTokens} の中だけ）

- 相対位置のバケット: 上流は `compute_bias` がグラフの中で `arange` → 差 → `log` → `where` で
  作る。ラッパはホストが作った表を入力で受け、層ごとの表 `[32, heads]` を `embedding` で引く
  （決定 3）。
- マスク: 上流は有効長で 1 の加算型マスク（`(1 − m)·finfo.min`）を全層のバイアスに足す。ラッパは
  持たない（決定 4）。
- FFN の活性: 上流は `gelu_new`（`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` の手書き式）。
  ラッパは `nn.GELU(approximate="tanh")`（同じ式の `aten.gelu` — 下の節）。
- dropout: 上流は eval で恒等。ラッパは持たない（{@link Umt5EncoderTokens} が eval を要求する）。

残りの部品（語彙埋め込み・q/k/v/o・RMSNorm・softmax を f32 で取る形・ゲート付き FFN・最後の
RMSNorm）は上流のモジュールをそのまま呼ぶか、上流の行を同じ順で書き下す。活性を上流のまま
（`activation="upstream"`）にしたラッパは上流の forward と f32 で**ビット一致**する（pytest が
縛る — 書き下しの誤りと活性の差し替えの丸めを分けて見るため）。

## `gelu_new` を差し替える理由（段 10c の下見の実測）

exporter は `gelu_new` の `pow(x, 3.0)` を拾えない（`aten.pow.Tensor_Scalar` が未対応 op の列挙に
落ちる — 正規化 `_pow2_to_mul` は指数 2 だけ）。`nn.GELU(approximate="tanh")` は数学的に同じ関数
（tanh 近似の GELU）を `aten.gelu` 1 本で書き、IR の `gelu_tanh` に降りる。差は f32 の丸めだけで、
要素ごとの実測は {@link wan.umt5_probe} の `activation_rounding`。

## バケット表（決定 3）

{@link relative_position_buckets} は上流の `UMT5Attention._relative_position_bucket` をそのまま呼ぶ
（式を写さない）。表の向きは上流の `compute_bias` と同じ `table[i][j] = bucket(j − i)`（キーの位置 −
クエリの位置）。TS の生成器（`packages/models/src/wan/umt5/relative-position.ts`）はこの表と
バイト一致する（fixture は {@link wan.umt5_host_fixture}）。

## 量子化の対象（決定 5）

`karume.quantize.fake_quant_int8` の既定の対象は `nn.Embedding` を含むので、層ごとの相対位置の表
（`relative_attention_bias` — `nn.Embedding [32, heads]`）まで i8 になる。{@link fake_quant_i8} は
include でこの表を外し、emit には {@link storage_overrides} で F32 を明示する（表は `embedding` の
重みスロットに入るので emit の適格集合に載り、scale が無いと `per-channel scale が無い` で落ちる）。

MUST: transformers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
from torch import nn
from torch.export import Dim
from torch.nn import functional

from karume.quantize import QUANT_MODULE_TYPES, Int8Report, fake_quant_int8, iter_quant_targets
from wan.umt5_tokenizer import MAX_LENGTH, MIN_TOKENS

#: グラフ入力の名前（並びは forward の引数の順 — TS の `src/wan/umt5/session-io.ts` と同じ綴り）。
INPUT_NAMES = ("input_ids", "relative_position_buckets")

#: 有効長の記号次元の名前（決定 4 — 1 シンボル）。
SYMBOL = "L"

#: 上流の FFN の活性（`config.dense_act_fn`）。これ以外の構成はラッパが扱わない。
UPSTREAM_ACTIVATION = "gelu_new"

#: ラッパの活性の選択（`tanh` = export の形・`upstream` = 書き下しの同値を見る形）。
Activation = Literal["tanh", "upstream"]

#: 相対位置の表のモジュール名の末尾（上流 `UMT5Attention` の属性名）。
RELATIVE_BIAS_ATTRIBUTE = "relative_attention_bias"


class Umt5ConfigError(ValueError):
    """ラッパが前提にする上流の構成（エンコーダ・ゲート付き `gelu_new`・層ごとの表）から外れた。"""


def check_supported(config: Any) -> None:
    """ラッパの書き下しが成り立つ構成かを見る（外れたら黙って別の計算をしない）。"""
    problems = []
    if config.is_decoder:
        problems.append("is_decoder が真（双方向のバケットと自己注意だけを書き下している）")
    if not config.is_gated_act:
        problems.append("FFN がゲート付きでない（`wi_0`・`wi_1`・`wo` の形だけを書き下している）")
    if config.dense_act_fn != UPSTREAM_ACTIVATION:
        problems.append(
            f"活性が {config.dense_act_fn!r}（{UPSTREAM_ACTIVATION!r} だけを差し替える）"
        )
    if problems:
        raise Umt5ConfigError("; ".join(problems))


def bucket_attention(config: Any) -> nn.Module:
    """バケットの式だけを持つ上流の `UMT5Attention`（重みは meta — 実体を確保しない）。

    実モデルを読まずに上流の式を呼ぶための口（fixture の生成 — {@link wan.umt5_host_fixture}）。
    q/k/v/o の Linear は meta 上に作るので、実寸の構成（4,096 × 4,096 × 4）でもメモリを取らない。
    """
    from transformers.models.umt5.modeling_umt5 import UMT5Attention

    check_supported(config)
    with torch.device("meta"):
        return UMT5Attention(config, has_relative_attention_bias=False)


def relative_position_buckets(length: int, attention: nn.Module) -> torch.Tensor:
    """有効長 `length` のバケット表 `[L, L]`（int64 — 上流の `compute_bias` の中間そのもの）。

    `attention` は上流の `UMT5Attention`（実モデルの層か {@link bucket_attention}）。上流の
    `compute_bias` の前半（位置の差 → `_relative_position_bucket`）と同じ行で、表の向きは
    `table[i][j] = bucket(j − i)`。`embedding` で層の表を引いて `[heads, L, L]` へ回すと上流の
    `compute_bias` とビット一致する（pytest）。

    受ける長さはグラフの記号次元と同じ 2〜512（決定 4）。TS の生成器と突き合わせたのもこの範囲
    だけ（f64 の `Math.log` と torch の f32 の `log` の一致は距離 ±511 の全件で実測 — 調査 §4.3）。
    境界での i32 への変換は呼び手が `karume.convert.normalize_boundary_tensor` で掛ける
    （ADR 0009）。
    """
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError(f"有効長は int で渡す（{type(length).__name__}）")
    if not MIN_TOKENS <= length <= MAX_LENGTH:
        raise ValueError(f"有効長 {length} が {MIN_TOKENS}〜{MAX_LENGTH} の外（決定 4）")
    if attention.is_decoder:
        raise Umt5ConfigError("デコーダの片方向バケットは扱わない（エンコーダの双方向だけ）")
    context_position = torch.arange(length, dtype=torch.long)[:, None]
    memory_position = torch.arange(length, dtype=torch.long)[None, :]
    return attention._relative_position_bucket(memory_position - context_position)


class Umt5EncoderTokens(nn.Module):
    """`UMT5EncoderModel` を「token id `[1, L]` + バケット表 `[L, L]` → `[1, L, d_model]`」にする。

    上流のモジュールは共有する（重みの丸め〈fake-quant〉はラッパ経由で上流にも届く）。
    `self.encoder` に上流の `UMT5Stack` だけを持たせるのは、tied な `shared` を名前の空間に入れない
    ため（重みのキーが `encoder.embed_tokens.weight` の 1 通りに決まる）。

    MUST: eval の上流だけを包む（dropout を持たないので、学習モードの上流とは別の計算になる）。
    MUST: 重みは f32（上流の FFN にある `wo` の手前の dtype 合わせを書き下していない）。
    """

    def __init__(self, model: nn.Module, *, activation: Activation = "tanh") -> None:
        super().__init__()
        check_supported(model.config)
        if model.training:
            raise Umt5ConfigError(
                "学習モードの上流は包まない（dropout が恒等でない）— eval() で渡す"
            )
        dtypes = {parameter.dtype for parameter in model.parameters()}
        if dtypes != {torch.float32}:
            raise Umt5ConfigError(f"重みの dtype {sorted(map(str, dtypes))} — f32 だけを包む")
        if activation not in ("tanh", "upstream"):
            raise ValueError(f"活性 {activation!r} は tanh / upstream のどちらか")
        self.encoder = model.encoder
        self.activation = activation
        self.gelu = nn.GELU(approximate="tanh")

    def forward(
        self, input_ids: torch.Tensor, relative_position_buckets: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder.embed_tokens(input_ids)
        for block in self.encoder.block:
            hidden = self._self_attention(block.layer[0], hidden, relative_position_buckets)
            hidden = self._feed_forward(block.layer[-1], hidden)
        return self.encoder.final_layer_norm(hidden)

    @staticmethod
    def _self_attention(
        layer: nn.Module, hidden: torch.Tensor, buckets: torch.Tensor
    ) -> torch.Tensor:
        """`UMT5LayerSelfAttention` + `UMT5Attention.forward` の書き下し（キャッシュ・マスクなし）。

        スコアにスケールを掛けない・softmax を f32 で取って元の dtype へ戻す、は上流の行のまま。
        バイアスだけが `compute_bias` の後半（表の gather → `[1, heads, L, L]`）で、表は入力から
        受ける。
        """
        attention = layer.SelfAttention
        normed = layer.layer_norm(hidden)
        batch_size, seq_length = normed.shape[:2]
        heads, width = attention.n_heads, attention.key_value_proj_dim
        query = attention.q(normed).view(batch_size, -1, heads, width).transpose(1, 2)
        key = attention.k(normed).view(batch_size, -1, heads, width).transpose(1, 2)
        value = attention.v(normed).view(batch_size, -1, heads, width).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(3, 2))
        bias = attention.relative_attention_bias(buckets).permute([2, 0, 1]).unsqueeze(0)
        scores = scores + bias
        weights = functional.softmax(scores.float(), dim=-1).type_as(scores)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length, -1)
        return hidden + attention.o(output)

    def _feed_forward(self, layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        """`UMT5LayerFF` + `UMT5DenseGatedActDense.forward` の書き下し（活性だけ選べる）。"""
        dense = layer.DenseReluDense
        normed = layer.layer_norm(hidden)
        activation = self.gelu if self.activation == "tanh" else dense.act
        gated = activation(dense.wi_0(normed)) * dense.wi_1(normed)
        return hidden + dense.wo(gated)


def dynamic_shapes() -> tuple[dict[int, Any], dict[int, Any]]:
    """入力ごとの記号次元。バケット表の 2 軸も token id と同じ `L` で宣言する（別シンボルにすると
    「表と本体の長さがずれた」形が受理されて沈黙誤値になる — DiT の RoPE の表と同じ理由）。"""
    length = Dim(SYMBOL, min=MIN_TOKENS, max=MAX_LENGTH)
    return ({1: length}, {0: length, 1: length})


def relative_bias_modules(wrapper: nn.Module) -> tuple[str, ...]:
    """層ごとの相対位置の表（`nn.Embedding [num_buckets, heads]`）のモジュール FQN。

    MUST: 本数が層数と一致する（umT5 は全層が自分の表を持つ — 調査 §1.1）。足りないと
    量子化から外し損ねた表が i8 で焼かれる。
    """
    names = tuple(
        name
        for name, module in wrapper.named_modules()
        if name.rsplit(".", 1)[-1] == RELATIVE_BIAS_ATTRIBUTE and isinstance(module, nn.Embedding)
    )
    layers = len(wrapper.encoder.block)
    if len(names) != layers:
        raise Umt5ConfigError(f"相対位置の表 {len(names)} 本が層数 {layers} と違う")
    return names


def quant_include(wrapper: nn.Module) -> Callable[[str], bool]:
    """i8 の対象の述語（モジュール FQN — 相対位置の表だけを外す）。"""
    tables = frozenset(relative_bias_modules(wrapper))
    return lambda name: name not in tables


def storage_overrides(wrapper: nn.Module) -> dict[str, str]:
    """emit へ渡す格納の明示（テンソルキー → `"f32"` — 相対位置の表を F32 のまま焼く）。

    表は `embedding` の重みスロットだけに消費されるので emit の適格集合に載る。既定 `i8` のまま
    scale が無いと emit が fail loudly で落ちるので、ここで F32 を明示する（決定 5）。
    """
    return {f"{name}.weight": "f32" for name in relative_bias_modules(wrapper)}


def fake_quant_i8(wrapper: nn.Module) -> Int8Report:
    """linear と語彙埋め込みの重みを per-channel symmetric i8 の表現可能値へ丸める（決定 5）。

    相対位置の表は丸めない（F32 のまま — 決定 5）。MUST: 丸めた本数が「量子化できる型の全重み −
    相対位置の表」と一致し、表の値が 1 ビットも動いていないことを確かめる — 外し損ねると golden と
    格納の両方が i8 の表になり、門では見えない。
    """
    include = quant_include(wrapper)
    tables = relative_bias_modules(wrapper)
    modules = dict(wrapper.named_modules())
    before = {name: modules[name].weight.detach().clone() for name in tables}
    expected = sorted(
        fqn
        for fqn, _, _ in iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES, include=include)
    )
    report = fake_quant_int8(wrapper, include=include)
    if sorted(report.scales) != expected:
        raise AssertionError(f"i8 に丸めた重みが想定と違う: {sorted(report.scales)} vs {expected}")
    moved = [name for name in tables if not torch.equal(modules[name].weight, before[name])]
    if moved:
        raise AssertionError(f"相対位置の表が丸められた: {moved}")
    return report


def padded_output(
    model: nn.Module, input_ids: torch.Tensor, max_length: int = MAX_LENGTH
) -> torch.Tensor:
    """上流の写し方（`_get_t5_prompt_embeds` — 512 まで pad id で詰めてマスク）の出力を
    有効長で切る。

    `model` は上流の `UMT5EncoderModel`（dtype は問わない — bf16 の比較にも使う）。マスクは上流の
    トークナイザと同じ int64 の 0 / 1。
    """
    length = int(input_ids.shape[1])
    if input_ids.shape[0] != 1 or not 0 < length <= max_length:
        raise ValueError(f"token id は [1, 1〜{max_length}] で渡す（{tuple(input_ids.shape)}）")
    padded = torch.full((1, max_length), model.config.pad_token_id, dtype=input_ids.dtype)
    padded[:, :length] = input_ids
    mask = torch.zeros((1, max_length), dtype=torch.long)
    mask[:, :length] = 1
    with torch.no_grad():
        output = model(input_ids=padded, attention_mask=mask).last_hidden_state
    return output[:, :length]


def valid_output(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """有効長だけで回した上流の出力（マスクを渡さない — 上流は全 1 のマスクを自分で作る）。"""
    with torch.no_grad():
        return model(input_ids=input_ids).last_hidden_state


def max_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """最大絶対差 ÷ 参照の最大絶対値（f64 で — `wan.export_dit` の帯の指標と同じ形）。"""
    expected = expected.double()
    return float((actual.double() - expected).abs().max() / expected.abs().max())


def compare(actual: torch.Tensor, expected: torch.Tensor) -> Mapping[str, Any]:
    """2 本の出力の一致の度合い（ビット一致・比・最大絶対差・違う要素の割合）。"""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"形か dtype が違う: {tuple(actual.shape)} {actual.dtype} vs "
            f"{tuple(expected.shape)} {expected.dtype}"
        )
    return {
        "bit_exact": bool(torch.equal(actual, expected)),
        "ratio": max_ratio(actual, expected),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
        "differing": float((actual != expected).double().mean()),
    }
