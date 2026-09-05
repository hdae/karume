"""RoPE バッファの lifted 検査 — モデル非依存の export 検証。

RoPE の周波数種（`inv_freq`）はバッファのままだと定数畳み込みの葉にならず、sin / cos が
IR 語彙に必要になる。素の属性（lifted tensor constant）へ降格して畳み込みの葉にするのが
{@link lift_rope_buffers} で、走査が空振りしたまま export へ進まないための門が
{@link assert_rope_lifted}。

対象は transformers 系の RoPE バッファ命名（接尾一致）で**特定のモデル / ファミリに依存
しない** — Anima の Qwen3 / EmbeddingGemma の Gemma3 / Irodori の text backbone が同じ形で
使う（ADR 0065 段 2 で Anima の patch 層から回収）。
"""

from __future__ import annotations

from torch import nn

#: RoPE の周波数種の**バッファ名接尾辞**。バッファのままだと定数畳み込みの葉にならず、
#: sin / cos が IR に残る。
#:
#: 完全一致ではなく接尾一致で見るのは、layer_type ごとに RoPE を分ける実装が接頭辞付きの
#: 名前を使うため（transformers 5.14.1 の Gemma3 は `sliding_attention_inv_freq` /
#: `full_attention_inv_freq` とその `original_*`）。既存の裸の名前は自身の接尾辞なので
#: 受理集合は真に広がるだけで、従来のモデルでのヒット本数は変わらない。
ROPE_BUFFER_NAMES = ("inv_freq", "original_inv_freq")


def lift_rope_buffers(root: nn.Module) -> int:
    """RoPE の `inv_freq` バッファを素の属性（lifted tensor constant）へ降格する。

    畳み込みの葉をパラメータ/バッファへ広げないのは巨大定数の焼き込みを避けるためだが
    （convert._classify_foldable）、`inv_freq` は head_dim/2 要素の位置表の種で、畳んだ結果
    （cos/sin 表）も Tmax × head_dim に収まる。降格しないと sin / cos が IR 語彙に必要になる。

    MUST: 1 本も降格できなければ呼び出し側が落とす（属性名が上流で変われば走査は静かに
    空振りし、以後どのモデルでも sin/cos が IR に残る — 恒真化の門は呼ぶ側に置く）。

    走査は {@link ROPE_BUFFER_NAMES} の**接尾一致**（layer_type 接頭辞付きの名前を持つ実装が
    ある）。`_buffers` を走査中に書き換えるので名前は先に採り切る。
    """
    lifted = 0
    for module in root.modules():
        for name in [key for key in module._buffers if key.endswith(ROPE_BUFFER_NAMES)]:
            tensor = module._buffers.pop(name)
            setattr(module, name, tensor.detach().clone())
            lifted += 1
    return lifted


def _rope_attribute_count(root: nn.Module) -> int:
    """RoPE の周波数種が載っている席の本数（バッファ席 + 降格済みの素の属性）。

    降格は `_buffers.pop` の破壊的操作なので、バッファ席だけを数えると**同じ module への
    2 度目**が必ず 0 本になる。`lift_rope_buffers` は pop の**後**に `setattr` するので
    `nn.Module.__setattr__` は buffer 席を見つけられず、降格済みの名前は素の `__dict__` に
    入る — 両方を数えれば「走査が空振りしている（上流で属性名が変わった）」の検出力は
    保ったまま冪等になる。判定規則は降格側と同じ接尾一致（{@link ROPE_BUFFER_NAMES}）。
    """
    return sum(
        1
        for module in root.modules()
        for names in (module._buffers, module.__dict__)
        for key in names
        if key.endswith(ROPE_BUFFER_NAMES)
    )


def assert_rope_lifted(root: nn.Module, where: str) -> None:
    """{@link lift_rope_buffers} を掛け、RoPE の周波数種が 1 本も無ければ落とす（恒真化の門）。

    `where` は失敗したときにどのモデル / コンポーネントの話かを示す呼び出し側の文脈。

    降格済みの module へ 2 度目を掛けても通る（数えるのは {@link _rope_attribute_count} =
    バッファ席と素の属性の総数）。「降格の戻り本数」で見ると 2 度目が必ず 0 本になり、
    事実と違う「上流で属性名が変わった」で export が止まる。
    """
    lift_rope_buffers(root)
    if _rope_attribute_count(root) == 0:
        raise ValueError(
            f"{where}: RoPE バッファ {ROPE_BUFFER_NAMES} が 1 本も見つからない"
            " — 降格の走査が空振りしている（上流で属性名が変わった可能性）。"
            "このまま export すると sin / cos が IR に残る"
        )
