"""実重み Gemma 4 E2B を **token-only 既定出口**（ADR 0068 決定 4）の states 形へ書き出す台本。

`export_decode.py`（logits opt-in 形 — 全 M 行に lm_head を通し `[logits, token]` を出す）の
**既定形**版。chunk 系列の経路（素材の読み方・states 手術・RoPE のホスト供給・混成量子化・
門の順序）は {@link gemma4.export_decode} の中核をそのまま通し、この台本が持つのはラッパ 1 つと
variant 記述（{@link VARIANT}）と入口だけ。出口の差は次の 3 点:

- 入力に **`last_row[1]` i32**（最終有効行の添字 = `queryLength − 1`）が増える。prefill
  チャンクの最終有効行は実行時スカラで、グラフから静的に切る手段が無い（ADR 0068 追記 3）—
  これを i32 グラフ入力で受けるのが決定 4 の想定した新配線。
- 出口は最終 norm 後の hidden `[1,M,H]` から **`F.embedding(last_row, hidden[0])`** で 1 行を
  選び（既存 `embedding` op — 添字が実行時値でも最終次元固定の行 gather なので新規 op 不要）、
  その 1 行だけに lm_head + softcap + argmax を通して **`token[1,1,1]` 1 本**を出す。
- 全語彙 logits はグラフ出力に**宣言しない**（readback は token 4B のみ。lm_head の計算も
  1 行ぶんに落ちる — prefill chunk あたり `(M−1) × 262144 × 1536` MAC の削減）。

    uv run --with 'transformers==5.14.1' python -m gemma4.export_token

## golden はこの系列では作らない（logits opt-in 系列との交差検証が門）

logits を出さない形の greedy 期待列は logits opt-in 系列（`gemma4-e2b-decode/`）の
`greedy.<case>.safetensors` と**同一のはず**（同じ重み・同じ丸め・同じ手術・出口の行選択が
同じ行を指す）。検収門（`e2e_gemma4_token_exit_test.ts`）はそれをそのまま流用して
「両系列の token 列が厳密一致」を見る — 系列間の交差検証そのものが門になり、torch 参照の
再計算（1 実走数十分）を払わない。本台本の sanity は各ケースの**第 1 継続 token** を
1-shot 台本の期待表と突き合わせるところまで（`export_decode` と同文）。

## 流用する golden は出所記録で束ねる

流用が成り立つ前提（**両系列が同じチェックポイントから出ている**）は、資産の存在確認では
守れない — 片方だけ古い組み合わせでも門は緑になる（全体レビュー CX-2.3）。そこで元
チェックポイントの指紋と、流用する golden 1 本ずつの digest を `reference.json` へ書き、
TS 側の検収門がそれを検めてから parity を見る（正本は {@link gemma4.provenance}）。
MUST: 記録を書くために logits opt-in 系列を**再 export しない** — 既存 golden を読むだけ。

## 出力レイアウト

    outputs/series/gemma4-e2b-decode-token/model.safetensors  重み・定数 + karume_ir
    outputs/series/gemma4-e2b-decode-token/reference.json     出所記録（指紋 + 流用 golden）
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional

from _shared.paths import SERIES_ROOT
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import ple

#: 生成物の既定の置き場（logits opt-in 系列とは別ディレクトリ — 出口の違う別資産）。
DEFAULT_OUT_DIR = SERIES_ROOT / "gemma4-e2b-decode-token"


class TokenOnlyChunkWrapper(decode.DecodeChunkWrapper):
    """`(input_ids, RoPE 4 本, last_row) → token[1,1,1]` の token-only chunk ラッパ。

    MUST: `DecodeChunkWrapper` の**派生**（モジュール FQN 空間の同一性 — 量子化の対象述語と
    scale 台帳の再利用条件。`export_decode.DecodeChunkWrapper` の docstring と同じ理由）。
    MUST: 行選択は `F.embedding`（`aten.embedding` → 既存 `embedding` op）。上流
    `Gemma4ForCausalLM.forward` の `logits_to_keep` にテンソルを渡す形は advanced indexing
    （`aten.index.Tensor` — IR 語彙外）に落ちるので使えない。lm_head と softcap の 3 行は
    上流 forward（modeling_gemma4.py:1895-1899）の鏡像 — 同じ config 値・同じ演算列なので
    選んだ行の数値は opt-in 形と一致する。
    """

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        rope_sliding_attention_cos: torch.Tensor,
        rope_sliding_attention_sin: torch.Tensor,
        rope_full_attention_cos: torch.Tensor,
        rope_full_attention_sin: torch.Tensor,
        last_row: torch.Tensor,
    ) -> torch.Tensor:
        length = input_ids.shape[1]
        mask = {
            one_shot.FULL_ATTENTION: one_shot.additive_causal_mask(length),
            one_shot.SLIDING_ATTENTION: one_shot.additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        stacked = ple.per_layer_inputs(self.per_layer, input_ids, self.per_layer_scale)
        tables = decode.bound_rope(
            rope_sliding_attention_cos,
            rope_sliding_attention_sin,
            rope_full_attention_cos,
            rope_full_attention_sin,
        )
        with self.model.model.rotary_emb.bound(tables):
            hidden = self.model.model(
                inputs_embeds=embeds,
                per_layer_inputs=stacked,
                attention_mask=mask,
                position_ids=None,
                use_cache=False,
            ).last_hidden_state
        # 行選択のあと [1,1,H] へ上げてから lm_head へ通す — argmax の**後ろ**に形合わせを
        # 置くと出力の供給元が reshape になり、「token 出力 = argmax 直結」（ADR 0068 決定 4 /
        # assert_ir_form_decode の検査）が崩れる。
        rowed = functional.embedding(last_row, hidden[0]).unsqueeze(0)
        logits = self.model.lm_head(rowed)
        cap = float(self.model.config.final_logit_softcapping)
        logits = torch.tanh(logits / cap) * cap
        return logits.argmax(-1, keepdim=True)


#: この台本の系列（token-only 既定出口）。経路は {@link gemma4.export_decode.export_series} が
#: そのまま持ち、ここが渡すのは 4 つの差分だけ。
#:
#: MUST: `goldens=False` — この系列は自分の greedy 記録を採らない（検収門が logits opt-in 系列の
#: `greedy.<case>.safetensors` を流用し、両系列の token 列が厳密一致することを門にする）。
VARIANT = decode.ChunkVariant(
    out_dir=DEFAULT_OUT_DIR, wrapper=TokenOnlyChunkWrapper, token_only=True, goldens=False
)


def main(argv: Sequence[str] | None = None) -> None:
    decode.run_variant_cli(VARIANT, __doc__.split("\n\n")[0], argv)


if __name__ == "__main__":
    main()
