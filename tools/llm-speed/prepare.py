"""公式tokenizerから速度比較の入力とローカル資産の対応を固定する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm-baseline"))
    from data import profiles

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=profiles(Path(".")), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--chat-template", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens は正の整数が必要")
    profile = profiles(Path.cwd())[args.model]
    if args.checkpoint is not None:
        profile["checkpoint"] = args.checkpoint.resolve()
    if args.source is not None:
        profile["series" if "series" in profile else "distribution"] = args.source.resolve()
    checkpoint = profile["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if args.chat_template is not None:
        tokenizer.chat_template = args.chat_template.read_text()
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError(
            "公式の会話テンプレートがありません。--chat-template で対応するファイルを指定してください"
        )
    generation = json.loads((checkpoint / "generation_config.json").read_text())
    stops = generation["eos_token_id"]
    stops = stops if isinstance(stops, list) else [stops]
    texts = {
        "english-list": "Write a numbered list of twenty practical tips for learning a new language. Give one short sentence for each tip.",
        "japanese-list": "プログラミングを学ぶ人に向けたアドバイスを20個、番号付きで挙げてください。それぞれ一文で説明してください。",
    }
    cases = []
    for name, prompt in texts.items():
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not ids or len(ids) + args.max_new_tokens - 1 > 128:
            raise ValueError(f"{name}: 入力と生成上限が容量128を超えます")
        cases.append({"case": name, "prompt": prompt, "inputIds": ids})
    result = {
        "format": "karume-llm-speed-input/2",
        "model": args.model,
        "profile": {k: str(v) if isinstance(v, Path) else v for k, v in profile.items()},
        "capacity": 128,
        "maxNewTokens": args.max_new_tokens,
        "stopTokens": stops,
        "cases": cases,
        "tokenizerSha256": hashlib.sha256((checkpoint / "tokenizer.json").read_bytes()).hexdigest(),
        "templateSha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "generationConfigSha256": hashlib.sha256(
            (checkpoint / "generation_config.json").read_bytes()
        ).hexdigest(),
    }
    args.out.mkdir()
    with (args.out / "inputs.json").open("x") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps({"model": args.model, "tokens": [len(c["inputIds"]) for c in cases]}))


if __name__ == "__main__":
    main()
