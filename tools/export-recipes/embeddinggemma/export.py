"""実重み EmbeddingGemma-300m（文埋め込み）を IR v1 コンテナ + golden io へ書き出す台本。

`deberta/export.py` と同じ役割（実重み・実トークン列での数値一致）を、SentenceTransformer 形の
**単一ベクトル出力**モデルで受け持つ。生成物は `outputs/series/` 配下で、リポジトリ直下の
`.gitignore` によりコミット対象外（重み 1.2GB 級）。

    uv run --with 'transformers==5.14.1' python -m embeddinggemma.export

transformers は **5.14.1 でピン**する（`deberta/export.py` と同じ理由 — モデリングコードが
変わるとグラフ形が変わる）。pyproject.toml / uv.lock には入れず `--with` で一時的に足す。

## 何をグラフに載せるか

SentenceTransformer の 5 モジュール（modules.json）のうち、**0〜4 の全段**を 1 グラフに載せる:

    Transformer(Gemma3TextModel) → Pooling(masked mean) → Dense(768→3072) → Dense(3072→768)
    → Normalize(L2)

出力は `[1, 768]` の単位ベクトル 1 本。プロンプト接頭辞の付与とトークナイズはホスト側
（`config_sentence_transformers.json` の `prompts` を逐語で使う）。

MUST: `Gemma3TextModel` は **`attention_mask` を渡さずに**呼ぶ。config が
`use_bidirectional_attention=true` なので、マスク無しなら sliding-window と全結合の 2 種類の
帯マスクが **T に依らない定数**として畳み込まれ（`sym_prefix_slice` で先頭 T を切り出す —
ADR 0010）、IR には mask 入力が 1 本も残らない。パディングを含む列を渡したいときは、呼び出し側
が列を詰めて T を短くする（この形は torch の eager と厳密に同値）。

MUST: SDPA は**保存する**（`PRESERVED_OP_PREFIXES_WITH_ATTENTION` — ADR 0023 の融合
attention）。台本ローカルの指定で、既定の分解表には入れない（ADR 0023 決定 5）。Gemma3 が
渡す bool の帯マスクは `normalize._additive_attn_mask` が加算型 f32 `[1,1,T,T]` へ落とし、
定数畳み込みで Tmax 定数 + `sym_prefix_slice` になる。分解経路へ落ちる道は**残さない** —
保存が通らなければ export ごと落ちるのが正しい（黙って別の数値経路に切り替わらない）。

MUST: 語彙外へ落ちる書き方を避ける。`torch.mean(dim=)` / `keepdim=True` / `x**2` /
`linalg_vector_norm` はいずれも IR 語彙に無い op（`mean.dim` / keepdim 付き reduce / `pow` /
`linalg_vector_norm`）を生むので、**mul / sum(dim) / reshape / sqrt / div / clamp_min** だけで
書く（`_masked_mean` と `_l2_normalize` の実装がその形）。

NOTE: `pool_mask` は全 1 で採る（マスク無しで呼ぶ以上、0 を混ぜると「モデルは見ているのに
プールでは捨てる」という eager 側にしか無い形になる）。入力として残すのは、ホストが列を
詰めきれない場合の逃げ道をランタイム側に残すため。

## 出力レイアウト

    outputs/series/embeddinggemma-300m/model.safetensors     重み・定数 + __metadata__.karume_ir
    outputs/series/embeddinggemma-300m/io.<case>.safetensors 入力と torch CPU での期待出力

io のテンソルキー規約は tiny golden / DeBERTa と同じ（`input.<グラフ入力名>` / `output.<位置>`）。

## batch 変種（`--batch`）

既定は `--batch 1`（本節冒頭の従来動作そのまま）。`--batch N`（N>1）は torch.export の
**静的次元**として batch を固定し、`query-en` を N 行に複製した単一 golden ケース
（`io.batchN.safetensors`）だけを書く — linear の GPU 時間が skinny-M（M=T）に律速されて
いる仮説を、M = batch×T を大きくして白黒つけるための資産（`--out` で置き場を明示する）。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.export import Dim

from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph
from karume.paths import INPUTS_ROOT, SERIES_ROOT
from karume.pipeline import export_to_file
from karume.rope import assert_rope_lifted

#: 公式重みの置き場（`hf download google/embeddinggemma-300m` の展開先）。
DEFAULT_MODEL_DIR = INPUTS_ROOT / "embeddinggemma" / "google-300m"

#: 生成物の既定の置き場。格納 dtype は f32 のみ（f16 / i8 は別系列で決める話）。
DEFAULT_OUT_DIR = SERIES_ROOT / "embeddinggemma-300m"

MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: SentenceTransformer の後段 2 段（modules.json の 2_Dense / 3_Dense）。
DENSE_DIRS = ("2_Dense", "3_Dense")
#: Dense の safetensors が持つ唯一のテンソルキー（bias 無しの `nn.Linear` 1 本）。
DENSE_WEIGHT_KEY = "linear.weight"

#: 記号次元 T の上限。sliding_window（512）と同値にする — 畳み込みの評価点そのものなので
#: 上げると帯マスク定数が Tmax² で膨らむ（ADR 0010）。
SYM_MAX = 512

#: masked mean の分母の下限（sentence-transformers の Pooling と同値）。
POOL_EPS = 1e-9
#: L2 正規化の分母の下限（sentence-transformers の Normalize と同値）。
NORM_EPS = 1e-12

#: 300 トークン級の長文（`long-document` ケースの本文）。Tmax 畳み込みの prefix スライスが
#: 実長で効いていることを、短文ケースと十分に離れた T で見るためのもの。
LONG_PASSAGE = (
    "Retrieval-augmented generation pipelines split a corpus into passages, embed each passage "
    "once, and keep the resulting vectors in an approximate nearest neighbour index. "
    "At query time the same encoder turns the user question into a vector, the index returns "
    "the closest passages, and a language model reads them before answering. "
    "The quality of such a system depends less on the generator than on the encoder: if the "
    "passage that actually answers the question is not in the top results, no amount of "
    "prompting will recover it. "
    "Running the encoder in the browser changes the trade-offs. There is no server to batch "
    "requests against, the weights have to be downloaded once and cached, and the compute "
    "budget is whatever the local GPU can spare between frames. "
    "A 300M parameter encoder in float32 is roughly 1.2 gigabytes, which is too much to fetch "
    "on every visit but perfectly reasonable to keep in a persistent cache. "
    "Quantising the weights to eight bits cuts that by a factor of four and, for retrieval, "
    "the loss in ranking quality is usually smaller than the loss from a mediocre chunking "
    "strategy. "
    "The interesting engineering problem is therefore not accuracy but scheduling: how to keep "
    "the embedding work off the main thread, how to overlap the download with the first few "
    "queries, and how to avoid re-encoding passages that have not changed since the last visit. "
    "Measured on a mid-range discrete GPU a single 512-token batch takes a few milliseconds "
    "once the pipelines are warm, so the practical ceiling is set by memory bandwidth rather "
    "than by arithmetic."
)

#: golden の固定文（`(ケース名, プロンプト種別 | None, 本文)`）。プロンプト文字列そのものは
#: `config_sentence_transformers.json` の `prompts` から**逐語で**引く（写しを持たない）。
#: 近い対（query-en × document-en）と遠い対（query-en × bare）を含めて、cosine の順序が
#: 意味と一致することを生成時に見られる並びにしてある。
GOLDEN_CASES: tuple[tuple[str, str | None, str], ...] = (
    ("query-en", "query", "what is the capital of France?"),
    ("document-en", "document", "Paris is the capital and most populous city of France."),
    ("query-ja", "query", "フランスの首都はどこですか。"),
    ("bare", None, "The quick brown fox jumps over the lazy dog."),
    ("long-document", "document", LONG_PASSAGE),
)

#: sanity 記録で見る cosine の対（近い対, 遠い対）。近い対が遠い対を上回らなければ、
#: 「数値は動いているが埋め込みとして壊れている」— 生成時に fail loudly にする。
NEAR_PAIR = ("query-en", "document-en")
FAR_PAIR = ("query-en", "bare")


def _masked_mean(hidden: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
    """`[1,T,H]` をトークン軸で masked mean する（`sentence_transformers` の Pooling と同値）。

    MUST: `keepdim=True` を使わない（IR の reduce は keepdim を持たない）。分母は
    `sum(dim=1)` の `[1]` を `unsqueeze` で `[1,1]` に戻してから broadcast 除算する
    （`unsqueeze` は正規化で `reshape` に落ちる）。
    """
    total = torch.sum(hidden * pool_mask.unsqueeze(-1), dim=1)
    count = torch.sum(pool_mask, dim=1).unsqueeze(-1)
    return total / torch.clamp(count, min=POOL_EPS)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """`[1,H]` を最終次元で L2 正規化する（`sentence_transformers` の Normalize と同値）。

    MUST: `x ** 2` でも `linalg.vector_norm` でもなく `x * x` + `sum(dim=-1)` + `sqrt` で書く
    （`pow` / `linalg_vector_norm` は IR 語彙に無い）。`clamp_min → 除算` の順序は原実装の
    まま — `+eps` に置き換えるとゼロ入力で 0 を返す性質が消える。
    """
    norm = torch.sqrt(torch.sum(x * x, dim=-1)).clamp(min=NORM_EPS).unsqueeze(-1)
    return x / norm


class EmbeddingWrapper(nn.Module):
    """SentenceTransformer の 5 段を 1 本の forward に畳んだ export 用ラッパ。

    MUST: `attention_mask` を渡さない（モジュール docstring の MUST）。渡すと加算バイアスが
    T 依存になり、帯マスクの畳み込み（`sym_prefix_slice`）が成立しなくなる。
    """

    def __init__(self, model: nn.Module, dense2: nn.Linear, dense3: nn.Linear) -> None:
        super().__init__()
        self.model = model
        self.dense2 = dense2
        self.dense3 = dense3

    def forward(self, input_ids: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, use_cache=False).last_hidden_state
        pooled = _masked_mean(hidden, pool_mask)
        return _l2_normalize(self.dense3(self.dense2(pooled)))


def load_dense(model_dir: Path, name: str) -> nn.Linear:
    """`<name>/config.json` の in/out と `<name>/model.safetensors` から bias 無し linear を組む。

    MUST: 形は config から読み、重みは safetensors から読む — どちらか一方から推測すると、
    上流が Dense の次元を変えたときに「読めたが別の層」が黙って通る。
    """
    config = json.loads((model_dir / name / "config.json").read_text(encoding="utf-8"))
    if config["bias"]:
        raise ValueError(f"{name}: bias 付きの Dense は想定外（公式配布は bias=false）")
    dense = nn.Linear(config["in_features"], config["out_features"], bias=False)
    weight = load_file(str(model_dir / name / MODEL_FILE))[DENSE_WEIGHT_KEY]
    if tuple(weight.shape) != tuple(dense.weight.shape):
        raise ValueError(
            f"{name}: 重み shape {tuple(weight.shape)} が config の"
            f" in/out から決まる {tuple(dense.weight.shape)} と食い違う"
        )
    dense.weight = nn.Parameter(weight.to(torch.float32))
    dense.eval()
    return dense


def load_wrapper(model_dir: Path) -> EmbeddingWrapper:
    """本体 + Dense 2 段を読み、RoPE バッファを降格した export 可能なラッパを返す。"""
    from transformers import Gemma3TextModel

    model = Gemma3TextModel.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    # inv_freq がバッファのままだと定数畳み込みの葉にならず、sin / cos が IR に残る。
    assert_rope_lifted(model, "embeddinggemma")
    dense2, dense3 = (load_dense(model_dir, name) for name in DENSE_DIRS)
    return EmbeddingWrapper(model, dense2, dense3).eval()


def load_prompts(model_dir: Path) -> dict[str, str]:
    """`config_sentence_transformers.json` の `prompts`（接頭辞の正本）。"""
    path = model_dir / "config_sentence_transformers.json"
    return json.loads(path.read_text(encoding="utf-8"))["prompts"]


def build_cases(
    model_dir: Path, sym_max: int
) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    """golden 5 ケースの `(名前, input_ids, pool_mask)`。

    トークナイズは `tokenizers` で公式 `tokenizer.json` を直接読む（bos / eos は tokenizer.json の
    post_processor が付ける — `tokenizer_config.json` の `add_bos_token` / `add_eos_token` が
    どちらも true であることと一致する）。
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompts = load_prompts(model_dir)
    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, prompt_key, body in GOLDEN_CASES:
        text = body if prompt_key is None else prompts[prompt_key] + body
        ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64)
        length = int(ids.shape[1])
        if not 2 <= length <= sym_max:
            raise ValueError(f"{name}: T={length} が記号次元の範囲 [2, {sym_max}] の外")
        # マスク無しで呼ぶ以上、プールで捨ててよいトークンは無い（モジュール docstring）。
        cases.append((name, ids, torch.ones_like(ids, dtype=torch.float32)))
    return tuple(cases)


def build_batch_case(
    model_dir: Path, sym_max: int, batch: int
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """`query-en` を `batch` 行に複製した単一ケース（M = batch×T を大きくする資産）。

    linear の GPU 時間が skinny-M（M=T=4〜318 で occupancy 不足）に律速されている仮説の
    白黒判定用。全行が同一文なので、torch 参照出力も全行同一になるはず — `_sanity_batch`
    で行間一致を検査する。GOLDEN_CASES から名前で引く（インデックス固定に依存しない）。
    """
    name, prompt_key, body = next(case for case in GOLDEN_CASES if case[0] == "query-en")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompts = load_prompts(model_dir)
    text = prompts[prompt_key] + body
    ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64)
    length = int(ids.shape[1])
    if not 2 <= length <= sym_max:
        raise ValueError(f"{name}: T={length} が記号次元の範囲 [2, {sym_max}] の外")
    batched_ids = ids.expand(batch, -1).contiguous()
    batched_mask = torch.ones_like(batched_ids, dtype=torch.float32)
    return f"batch{batch}", batched_ids, batched_mask


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    戻り値の 2 本目は sanity 記録用の期待出力（`[B, H]` の形のまま渡す — batch>1 の
    行間比較に形が要る。batch=1 では `_sanity` 側で `reshape(-1)` して従来どおり扱う）。
    """
    written: list[str] = []
    embeddings: dict[str, torch.Tensor] = {}
    if len(graph.outputs) != 1:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（埋め込みは 1 本）")
    for name, ids, pool_mask in cases:
        with torch.no_grad():
            output = wrapper(ids, pool_mask)
        args = {"input_ids": ids, "pool_mask": pool_mask}
        # MUST: io も IR の意味論 dtype の実表現へ落とす（i64 → i32）。ランタイムが受け取る
        # 形と揃っていないと Deno 側 E2E が golden を読めない（ADR 0009 の境界正規化）。
        tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name} の入力 '{declared.name}'"
            )
            for declared in graph.inputs
        }
        tensors[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
            output.detach().contiguous(), f"{name} の出力 0"
        )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        embeddings[name] = output.detach()
    return written, embeddings


def _sanity(embeddings: dict[str, torch.Tensor]) -> dict[str, Any]:
    """出力が単位ベクトルで、意味の近い対の cosine が遠い対を上回ることを見る（batch=1 専用）。

    MUST: 順序が逆なら落とす。ノルムだけでは「正規化は効いているが埋め込みが壊れている」
    （層の取り違え・プールの軸違い）を検出できない — 恒真な sanity にしない。
    """
    vectors = {name: output.reshape(-1) for name, output in embeddings.items()}
    norms = {name: float(vector.norm()) for name, vector in vectors.items()}
    off = [name for name, norm in norms.items() if abs(norm - 1.0) > 1e-5]
    if off:
        raise AssertionError(f"L2 ノルムが 1 から外れたケース: {[(n, norms[n]) for n in off]}")

    def cosine(pair: tuple[str, str]) -> float:
        return float(torch.dot(vectors[pair[0]], vectors[pair[1]]))

    near, far = cosine(NEAR_PAIR), cosine(FAR_PAIR)
    if near <= far:
        raise AssertionError(
            f"cosine の順序が意味と逆: {NEAR_PAIR}={near:.4f} <= {FAR_PAIR}={far:.4f}"
        )
    return {
        "l2_norms": {name: round(norm, 7) for name, norm in norms.items()},
        "cosine": {
            f"{NEAR_PAIR[0]}×{NEAR_PAIR[1]}": round(near, 4),
            f"{FAR_PAIR[0]}×{FAR_PAIR[1]}": round(far, 4),
            "query-en×query-ja": round(cosine(("query-en", "query-ja")), 4),
        },
    }


#: 行間比較の許容誤差（batch 内は同一入力なので理論上は完全一致 — 浮動小数の丸め分だけ許す）。
BATCH_ROW_ATOL = 1e-4


def _sanity_batch(output: torch.Tensor) -> dict[str, Any]:
    """batch 変種の sanity（batch>1 専用）: 全行が単位ベクトルで、複製元の行と一致することを見る。

    MUST: 行間の不一致を見逃さない — 全行が同一入力の複製である以上、崩れていれば
    「重みは合っているが batch 軸の扱いが壊れている」ことを意味する（`_sanity` の cosine
    順序検査に相当する、batch 変種向けの恒真でない検査）。
    """
    norms = output.norm(dim=-1)
    off = (norms - 1.0).abs() > 1e-5
    if bool(off.any()):
        raise AssertionError(f"L2 ノルムが 1 から外れた行あり: {norms[off].tolist()}")

    max_row_diff = float((output - output[0]).abs().max())
    if max_row_diff > BATCH_ROW_ATOL:
        raise AssertionError(
            f"行間の出力が一致しない（最大絶対差 {max_row_diff} > {BATCH_ROW_ATOL}）"
            " — batch 軸の扱いが壊れている"
        )
    return {
        "rows": int(output.shape[0]),
        "l2_norms_min_max": (round(float(norms.min()), 7), round(float(norms.max()), 7)),
        "row_max_abs_diff": round(max_row_diff, 9),
    }


def export_series(
    model_dir: Path, out_dir: Path, *, sym_max: int = SYM_MAX, batch: int = 1
) -> dict[str, Any]:
    """IR コンテナと golden io を書き、要約を返す。

    `batch` は torch.export の**静的次元**（既定 1 = 従来どおり全 5 ケース）。T は
    従来どおり動的次元のまま。`batch > 1` のときは `query-en` を `batch` 行に複製した
    単一ケースだけを golden にする（linear の occupancy 不足仮説の検証用資産）。
    """
    wrapper = load_wrapper(model_dir)
    cases = (
        build_cases(model_dir, sym_max)
        if batch == 1
        else (build_batch_case(model_dir, sym_max, batch),)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 例示入力は最長ケース（T が上限に近いほど 0/1 特殊化から遠い）。min=2 は 0/1 特殊化を
    # 避けるため、max は Tmax 畳み込みの評価点そのもの（ADR 0010 — 別ノブで二重管理しない）。
    _, example_ids, example_mask = max(cases, key=lambda case: case[1].shape[1])
    seq = Dim("T", min=2, max=sym_max)
    graph = export_to_file(
        wrapper,
        (example_ids, example_mask),
        out_dir / MODEL_FILE,
        dynamic_shapes=({1: seq}, {1: seq}),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    written, embeddings = _write_io(wrapper, graph, cases, out_dir)
    sanity = _sanity(embeddings) if batch == 1 else _sanity_batch(next(iter(embeddings.values())))
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": (out_dir / MODEL_FILE).stat().st_size,
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "case_lengths": {name: int(ids.shape[1]) for name, ids, _ in cases},
        "batch": batch,
        "sanity": sanity,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="torch.export の静的 batch 次元（既定 1 = 従来どおり）。1 より大きいと"
        " query-en を batch 行に複製した単一 golden ケースだけを書く。",
    )
    args = parser.parse_args(argv)
    if args.batch < 1:
        raise SystemExit(f"--batch は 1 以上を指定する（指定は {args.batch}）")
    summary = export_series(args.model_dir, args.out, sym_max=args.sym_max, batch=args.batch)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
