"""保存重みを公式Transformersで生成し、暖機とTTFTを分けて測る。"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import torch
import transformers
from torch import nn
from transformers import Gemma4ForCausalLM


class DevicePle(nn.Module):
    """通常Gemmaの巨大PLEはCPUの同一行を読み、使う行だけdeviceへ運ぶ。"""

    def __init__(self, rows: nn.Module, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.rows = rows
        self.target = device
        self.output_dtype = dtype

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.rows(ids.cpu()).to(device=self.target, dtype=self.output_dtype)


def save(path: Path, value: object) -> None:
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def timing(start: float, first: float | None, end: float, count: int) -> dict:
    return {
        "elapsedMs": (end - start) * 1000,
        "generatedTokens": count,
        "ttftMs": None if first is None else (first - start) * 1000,
        "decodeTokensPerSecond": None
        if first is None or count < 2 or end <= first
        else (count - 1) / (end - first),
    }


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm-baseline"))
    from run import stored_paths
    from weights import StoredWeights, fingerprint, load_float_model, load_qat

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", choices=["stored", "source"], default="stored")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--qat-model", choices=["text", "conditional"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.repeats, args.warmups, args.threads) < 1:
        raise ValueError("repeats/warmups/threadsは正の整数が必要")
    if transformers.__version__ != "5.14.1":
        raise ValueError("Transformers 5.14.1で検証する")
    fixture = json.loads(args.inputs.read_text())
    if fixture["format"] != "karume-llm-speed-input/2":
        raise ValueError("未知の入力形式")
    name = fixture["model"]
    profile = {
        k: Path(v) if k in ("checkpoint", "series", "distribution") else v
        for k, v in fixture["profile"].items()
    }
    qat = profile["family"] == "gemma4-qat"
    if not qat and args.qat_model is not None:
        raise ValueError("qat-model はQATだけの条件です")
    qat_model = args.qat_model or "text"
    if qat and (args.dtype != "float32" or args.weights != "stored"):
        raise ValueError("QATは固定重み・SRQを保つstored/float32で測る。半精度へ暗黙に変更しない。")
    args.out.mkdir()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDAが利用できません")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPSが利用できません")

    def sync() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "mps":
            torch.mps.synchronize()

    data = fixture
    if fingerprint(profile["checkpoint"] / "tokenizer.json")["sha256"] != data["tokenizerSha256"]:
        raise ValueError("固定したtokenizerと異なります")
    max_new = fixture["maxNewTokens"]
    stop_tokens = data["stopTokens"]
    if fixture["capacity"] != 128 or type(max_new) is not int or max_new < 1:
        raise ValueError("この比較は容量128・正の生成上限が必要です")
    if not data["cases"] or len({c["case"] for c in data["cases"]}) != len(data["cases"]):
        raise ValueError("評価ケースは空でない一意な名前が必要です")
    for case in data["cases"]:
        if not re.fullmatch(r"[a-z0-9-]+", case["case"]) or not case["inputIds"]:
            raise ValueError("不正なケース名または空の入力です")
    all_ids = [*stop_tokens, *(i for c in data["cases"] for i in c["inputIds"])]
    if any(type(i) is not int or i < 0 for i in all_ids) or not stop_tokens:
        raise ValueError("token ID は非負整数で、停止集合が必要です")
    if any(len(c["inputIds"]) + max_new - 1 > fixture["capacity"] for c in data["cases"]):
        raise ValueError("固定入力が容量を超える")
    started = time.perf_counter()
    shards = []
    ple = None
    stored = None
    if args.weights == "stored":
        shards, ple = stored_paths(profile)
        stored = StoredWeights(shards)
    if qat:
        model, checks = load_qat(profile["checkpoint"], stored, ple)
        if qat_model == "text":
            # 同じ公式量子化層を使い、テキストに不要なmultimodalの表全体展開を避ける。
            with torch.device("meta"):
                text_model = Gemma4ForCausalLM(model.config.get_text_config())
            text_model.model = model.model.language_model
            text_model.lm_head = model.lm_head
            model = text_model
            model.set_attn_implementation(args.attention)
        else:
            model.model.language_model.set_attn_implementation(args.attention)
    else:
        model, checks = load_float_model(profile["checkpoint"], profile["family"], stored, ple)
        # 復元時の同値なParameterを公式の共有参照へ戻してからdeviceへ転送する。
        model.tie_weights()
        model.set_attn_implementation(args.attention)
        if profile["family"] == "gemma4":
            model.model.embed_tokens_per_layer = DevicePle(
                model.model.embed_tokens_per_layer, device, dtype
            )
    vocab = model.config.get_text_config().vocab_size
    if any(i >= vocab for i in all_ids):
        raise ValueError("token ID が語彙の範囲外です")
    parameter_bytes = sum(
        p.numel()
        * (
            torch.empty((), dtype=dtype).element_size()
            if p.is_floating_point()
            else p.element_size()
        )
        for p in model.parameters()
    )
    if args.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        if parameter_bytes > free:
            save(
                args.out / "capacity.json",
                {"parameterBytes": parameter_bytes, "freeBytes": free, "totalBytes": total},
            )
            raise ValueError(
                "このdtypeの重みがGPUに収まりません。別dtypeは別条件として明示して測ってください。"
            )
    model = model.to(device=device, dtype=dtype).eval()
    sync()
    loaded = time.perf_counter() - started
    files = [
        fingerprint(profile["checkpoint"] / "config.json"),
        fingerprint(profile["checkpoint"] / "tokenizer.json"),
    ]
    files.extend(fingerprint(p) for p in shards)
    if ple is not None:
        files.append(fingerprint(ple))
        files.extend(
            fingerprint(ple.parent / s["file"]) for s in json.loads(ple.read_text())["shards"]
        )
    if args.weights == "source" or qat:
        files.extend(fingerprint(p) for p in sorted(profile["checkpoint"].glob("*.safetensors")))
    save(
        args.out / "inputs.json",
        {
            "fixture": fingerprint(args.inputs),
            "weights": files,
            "script": fingerprint(Path(__file__)),
            "loaderCode": [
                fingerprint(Path(__file__).resolve().parents[1] / "llm-baseline" / n)
                for n in ["weights.py", "data.py", "run.py"]
            ],
        },
    )

    def generate(ids: list[int]) -> dict:
        sync()
        first = None
        tokens = []
        past = None
        stop = None
        current = ids
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(max_new):
                input_ids = torch.tensor([current], dtype=torch.int64, device=device)
                output = model(
                    input_ids=input_ids, past_key_values=past, use_cache=True, logits_to_keep=1
                )
                token = int(output.logits[0, -1].argmax().item())
                past = output.past_key_values
                if token in stop_tokens:
                    stop = token
                    break
                if first is None:
                    first = time.perf_counter()
                tokens.append(token)
                current = [token]
        sync()
        ended = time.perf_counter()
        if not bool(torch.isfinite(output.logits).all().item()):
            raise ValueError("生成結果に非有限のlogitsがあります")
        record = {
            **timing(start, first, ended, len(tokens)),
            "tokenIds": tokens,
            "stopToken": stop,
            "stopReason": "max-tokens" if stop is None else "eos",
            "promptTokens": len(ids),
        }
        del past
        return record

    records = []
    for case_index, case in enumerate(data["cases"]):
        first = generate(case["inputIds"])
        row = {
            "case": case["case"],
            "first": first,
            "firstInProcess": case_index == 0,
            "warmups": [],
            "measured": [],
        }
        save(args.out / f"{case['case']}-first.json", first)
        print(case["case"], "first", first["ttftMs"], first["decodeTokensPerSecond"], flush=True)
        for i in range(args.warmups):
            value = generate(case["inputIds"])
            row["warmups"].append(value)
            save(args.out / f"{case['case']}-warmup-{i}.json", value)
        for i in range(args.repeats):
            value = generate(case["inputIds"])
            row["measured"].append(value)
            save(args.out / f"{case['case']}-measured-{i}.json", value)
            print(case["case"], i, value["ttftMs"], value["decodeTokensPerSecond"], flush=True)
        assert all(
            r["tokenIds"] == first["tokenIds"] and r["stopToken"] == first["stopToken"]
            for r in row["warmups"] + row["measured"]
        )
        records.append(row)
    summary = {
        "format": "karume-llm-speed-result/1",
        "engine": "transformers",
        "model": name,
        "modelClass": type(model).__name__,
        "qatModel": qat_model if qat else None,
        "weights": args.weights,
        "device": args.device,
        "deviceName": torch.cuda.get_device_name() if args.device == "cuda" else platform.machine(),
        "dtype": args.dtype,
        "attention": args.attention,
        "float32MatmulPrecision": torch.get_float32_matmul_precision(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "threads": args.threads,
        "loadSeconds": loaded,
        "parameterBytes": parameter_bytes,
        "cudaPeakAllocatedBytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "placement": "official-packed-qat" if qat else "dequantized-dense",
        "plePlacement": "cpu-row-lookup"
        if profile["family"] == "gemma4"
        else "device-packed"
        if qat
        else None,
        "capacity": fixture["capacity"],
        "maxNewTokens": max_new,
        "cache": "Transformers default dynamic",
        "compile": False,
        "textDecodingTimed": False,
        "loaderChecks": checks,
        "cases": records,
    }
    save(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
