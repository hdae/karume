"""実重み Gemma 4 E2B を **token-only 既定出口**（ADR 0068 決定 4）の states 形へ書き出す台本。

`export_decode.py`（logits opt-in 形 — 全 M 行に lm_head を通し `[logits, token]` を出す）の
**既定形**版。decode 系列の本体（states 手術・RoPE 表引き・混成量子化）はそのまま再利用し、
出口だけを差し替える:

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

## 出力レイアウト

    outputs/series/gemma4-e2b-decode-token/model.safetensors  重み・定数 + karume_ir
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.export import Dim
from torch.nn import functional

from _shared.paths import SERIES_ROOT
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from karume.artifacts import staged_publication
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.pipeline import export_module
from karume.states import to_states_form

#: 生成物の既定の置き場（logits opt-in 系列とは別ディレクトリ — 出口の違う別資産）。
DEFAULT_OUT_DIR = SERIES_ROOT / "gemma4-e2b-decode-token"

DEFAULT_MODEL_DIR = one_shot.DEFAULT_MODEL_DIR


class TokenOnlyChunkWrapper(decode.DecodeChunkWrapper):
    """`(input_ids, position_ids, last_row) → token[1,1,1]` の token-only chunk ラッパ。

    MUST: `DecodeChunkWrapper` の**派生**（モジュール FQN 空間の同一性 — 量子化の対象述語と
    scale 台帳の再利用条件。`export_decode.DecodeChunkWrapper` の docstring と同じ理由）。
    MUST: 行選択は `F.embedding`（`aten.embedding` → 既存 `embedding` op）。上流
    `Gemma4ForCausalLM.forward` の `logits_to_keep` にテンソルを渡す形は advanced indexing
    （`aten.index.Tensor` — IR 語彙外）に落ちるので使えない。lm_head と softcap の 3 行は
    上流 forward（modeling_gemma4.py:1895-1899）の鏡像 — 同じ config 値・同じ演算列なので
    選んだ行の数値は opt-in 形と一致する。
    """

    def forward(  # type: ignore[override]
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, last_row: torch.Tensor
    ) -> torch.Tensor:
        length = input_ids.shape[1]
        mask = {
            one_shot.FULL_ATTENTION: one_shot.additive_causal_mask(length),
            one_shot.SLIDING_ATTENTION: one_shot.additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        stacked = one_shot.per_layer_inputs(self.per_layer, input_ids, self.per_layer_scale)
        hidden = self.model.model(
            inputs_embeds=embeds,
            per_layer_inputs=stacked,
            attention_mask=mask,
            position_ids=position_ids,
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


def load_wrapper(
    model_dir: Path, *, positions: int = decode.ROPE_TABLE_POSITIONS
) -> TokenOnlyChunkWrapper:
    """実重みを f32 で読み、RoPE を表引きへ差し替えた token-only ラッパを返す。

    素材の読み方は `export_decode.load_wrapper` と同一（検査席の PLE 表を落とす理由も同文）—
    違うのは最後に組むラッパ型だけ。
    """
    model, tables = one_shot.load_model_and_tables(model_dir)
    del model.model.embed_tokens_per_layer
    decode.swap_rope_table(model, positions)
    return TokenOnlyChunkWrapper(model, tables).eval()


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = one_shot.SYM_MAX,
    positions: int = decode.ROPE_TABLE_POSITIONS,
) -> dict[str, Any]:
    """token-only の IR コンテナを書き、要約を返す。

    MUST: 公開は形検査と sanity の**後**（`export_decode._publish` と同じ理由の単一ファイル版
    — 門より前に置くと落ちた実走が検収を通れる資産を残す）。門を作業席の中に置く形そのものは
    core の原語（{@link karume.artifacts.staged_publication}）が持つ。
    """
    wrapper = load_wrapper(model_dir, positions=positions)
    # MUST: 丸めは参照・golden の採取より前（ADR 0006）。
    int8, int4, scales = one_shot.quantize_wrapper(wrapper)
    cases = one_shot.build_cases(model_dir, sym_max, wrapper.sliding_window)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, example_ids = max(cases, key=lambda case: case[1].shape[1])
    example_row = torch.tensor([int(example_ids.shape[1]) - 1], dtype=torch.int64)
    seq = Dim(decode.SEQ_SYMBOL, min=2, max=sym_max)
    print("[export] torch.export → 変換", file=sys.stderr, flush=True)
    graph, tensors = export_module(
        wrapper,
        (example_ids, decode.positions_for(example_ids), example_row),
        # last_row は静的 `[1]`（動的次元なし）。
        dynamic_shapes=({1: seq}, {1: seq}, None),
        symbol_names=(decode.SEQ_SYMBOL,),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    config = wrapper.model.config
    print("[export] states 形へ手術 → 書き出し", file=sys.stderr, flush=True)
    surgical = to_states_form(graph, decode.states_plan(graph, config))
    final = out_dir / one_shot.MODEL_FILE
    with staged_publication(final) as staged:
        verified = decode._write_container(
            surgical,
            tensors,
            staged,
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides={
                **dict.fromkeys(int4.scales, "i4"),
                **dict.fromkeys(decode.rope_table_keys(wrapper), "f32"),
            },
        )
        form = decode.assert_ir_form_decode(
            verified, config, {"i8": len(int8.scales), "i4": len(int4.scales)}, token_only=True
        )

        # sanity: 各ケースの第 1 継続 token（全長 1 chunk + last_row = T−1）が 1-shot 台本の
        # 期待表と一致する（`export_decode` の交差 sanity と同文 — 公開より前に評価する）。
        print("[sanity] 全長 forward", file=sys.stderr, flush=True)
        first: dict[str, int] = {}
        for name, ids in cases:
            row = torch.tensor([int(ids.shape[1]) - 1], dtype=torch.int64)
            with torch.no_grad():
                token = wrapper(ids, decode.positions_for(ids), row)
            first[name] = int(token[0, 0, 0])
        tokenizer = one_shot.load_tokenizer(model_dir)
        expected = one_shot.expected_token_ids(tokenizer)
        labels = {
            token: tokenizer.id_to_token(token)
            for token in set(first.values()) | set(expected.values())
        }
        sanity = one_shot._sanity(first, expected, labels)
    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "model_bytes": final.stat().st_size,
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "quantized": {"i8": int8.describe(), "i4": int4.describe()},
        "form": form,
        "sanity": sanity,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sym-max", type=int, default=one_shot.SYM_MAX)
    parser.add_argument("--positions", type=int, default=decode.ROPE_TABLE_POSITIONS)
    args = parser.parse_args(argv)
    summary = export_series(
        args.model_dir, args.out, sym_max=args.sym_max, positions=args.positions
    )
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
