"""PyTorch/Transformers の品質ベースライン。既存出力を上書きしない。"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import resource
import time
from pathlib import Path

import torch
import transformers
from data import profiles
from karume.shards import resolve_shards
from karume.verify import verify_model
from safetensors.torch import load_file
from scoring import accuracy, token_nll, windows
from weights import StoredWeights, fingerprint, load_float_model, load_qat


def stored_paths(profile: dict) -> tuple[list[Path], Path | None]:
    if "series" in profile:
        path = profile["series"] / "model.safetensors"
        verify_model(path)
        return list(resolve_shards(path)), None
    root = profile["distribution"]
    manifest = json.loads((root / "karume.json").read_text())
    if manifest["format"] != "karume/4":
        raise ValueError("karume/4 の配布形が必要")
    model = manifest["models"][profile["model"]]
    declarations = model["weights"]["model"]["i4"]["shards"]
    if any("repo" in shard for shard in declarations):
        raise ValueError("越境参照はこのローカル評価では未対応")
    shards = [root / shard["path"] for shard in declarations]
    virtual = re.sub(r"-\d{5}-of-\d{5}(?=\.safetensors$)", "", str(shards[0]))
    verify_model(Path(virtual))
    ple = [
        root / value["path"]
        for value in model["assets"].values()
        if value["path"].endswith("/ple.json") or value["path"] == "ple.json"
    ]
    if len(ple) != 1:
        raise ValueError("PLE索引が一意ではありません")
    return shards, ple[0]


def save(path: Path, value: object) -> None:
    with path.open("x") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model", choices=profiles(Path(".")), required=True)
    parser.add_argument("--weights", choices=["source", "stored"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--arc-limit", type=int)
    parser.add_argument("--wiki-limit", type=int)
    args = parser.parse_args()
    if transformers.__version__ != "5.14.1":
        raise ValueError("この評価は Transformers 5.14.1 で検証しています")
    if args.threads < 1 or any(n is not None and n < 1 for n in [args.arc_limit, args.wiki_limit]):
        raise ValueError("件数とthreadsは正である必要があります")
    args.out.mkdir()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    suite = json.loads(args.suite.read_text())
    if suite["format"] != "karume-llm-quality/1":
        raise ValueError("不明な評価集合")
    data = suite["models"][args.model]
    profile = profiles(args.root)[args.model]
    checkpoint = profile["checkpoint"]
    tokenizer = fingerprint(checkpoint / "tokenizer.json")
    if tokenizer["sha256"] != data["tokenizerSha256"]:
        raise ValueError("固定したtokenizerと現在のモデルが一致しません")
    if profile["family"] == "gemma4-qat" and args.weights != "stored":
        raise ValueError("QATは公式の固定量子化とstoredが同じ1構成です。storedで実行してください")
    at = time.perf_counter()
    files = [fingerprint(path) for path in sorted(checkpoint.glob("*.safetensors"))]
    files += [fingerprint(checkpoint / "config.json"), tokenizer]
    stored = None
    ple = None
    if args.weights == "stored":
        shards, ple = stored_paths(profile)
        files += [fingerprint(path) for path in shards]
        if ple is not None:
            files.append(fingerprint(ple))
            files += [
                fingerprint(ple.parent / shard["file"])
                for shard in json.loads(ple.read_text())["shards"]
            ]
        stored = StoredWeights(shards)
    save(
        args.out / "inputs.json",
        {
            "suite": fingerprint(args.suite),
            "files": files,
            "model": args.model,
            "weights": args.weights,
            "code": [fingerprint(p) for p in sorted(Path(__file__).parent.glob("*.py"))],
        },
    )
    if profile["family"] == "gemma4-qat":
        if stored is None or ple is None:
            raise ValueError("QATの固定重みとPLEが必要")
        model, checks = load_qat(checkpoint, stored, ple)
    else:
        model, checks = load_float_model(checkpoint, profile["family"], stored, ple)
    if args.weights == "stored" and "series" in profile:
        fixture_path = profile["series"] / "io.capital-en.safetensors"
        fixture = load_file(fixture_path)
        with torch.inference_mode():
            actual = model(input_ids=fixture["input.input_ids"].long(), use_cache=False).logits
        expected = fixture["output.0"]
        # 既存の公式モデルへの復元検証と同じ絶対誤差の門。期待値は変更しない。
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=0)
        checks["fixture"] = {
            **fingerprint(fixture_path),
            "maxAbs": float((actual - expected).abs().max()),
            "atol": 1e-3,
        }
    save(args.out / "loader.json", checks)
    print(
        json.dumps(
            {
                "loadedSeconds": time.perf_counter() - at,
                "model": args.model,
                "weights": args.weights,
            }
        ),
        flush=True,
    )

    def logits(ids: list[int]) -> torch.Tensor:
        with torch.inference_mode():
            return model(input_ids=torch.tensor([ids], dtype=torch.int64), use_cache=False).logits

    arc = []
    for question in data["arc"][: args.arc_limit]:
        started = time.perf_counter()
        choices = []
        for ids in question["choices"]:
            nll = token_nll(logits(ids), ids, question["scoreStart"])
            choices.append({"sumLogProb": -math.fsum(nll), "tokenCount": len(nll), "tokenNll": nll})
        predicted = max(range(len(choices)), key=lambda i: choices[i]["sumLogProb"])
        normalized = max(
            range(len(choices)), key=lambda i: choices[i]["sumLogProb"] / choices[i]["tokenCount"]
        )
        result = {
            "id": question["id"],
            "answer": question["answer"],
            "predicted": predicted,
            "tokenNormalizedPredicted": normalized,
            "choices": choices,
            "seconds": time.perf_counter() - started,
        }
        arc.append(result)
        save(args.out / f"arc-{len(arc):03}.json", result)
        print(
            "arc",
            len(arc),
            predicted == question["answer"],
            round(result["seconds"], 2),
            flush=True,
        )
    wiki_ids = data["wikiIds"]
    wiki = []
    window_count = 0
    for start, end, score_start in windows(
        len(wiki_ids), suite["wiki"]["window"], suite["wiki"]["stride"]
    ):
        if args.wiki_limit is not None and window_count >= args.wiki_limit:
            break
        ids = wiki_ids[start:end]
        nll = token_nll(logits(ids), ids, score_start)
        record = {"start": start, "end": end, "scoreStart": score_start, "tokenNll": nll}
        wiki.extend(nll)
        window_count += 1
        save(args.out / f"wiki-{window_count:03}.json", record)
        print("wiki", window_count, len(wiki), flush=True)
    if not arc or not wiki:
        raise ValueError("評価結果がありません")
    if args.wiki_limit is None and len(wiki) != len(wiki_ids) - 1:
        raise ValueError("WikiTextの重複・未採点tokenがあります")
    nll = math.fsum(wiki) / len(wiki)
    summary = {
        "format": "karume-llm-quality-result/1",
        "model": args.model,
        "weights": args.weights,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torchBuild": torch.__config__.show(),
        "device": "cpu",
        "dtype": "float32",
        "attention": "eager",
        "threads": args.threads,
        "arc": accuracy(sum(r["predicted"] == r["answer"] for r in arc), len(arc)),
        "arcTokenNormalized": accuracy(
            sum(r["tokenNormalizedPredicted"] == r["answer"] for r in arc), len(arc)
        ),
        "wiki": {
            "tokens": len(wiki),
            "windows": window_count,
            "meanNll": nll,
            "perplexity": math.exp(nll),
        },
        "partial": args.arc_limit is not None or args.wiki_limit is not None,
        "seconds": time.perf_counter() - at,
        "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    save(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
