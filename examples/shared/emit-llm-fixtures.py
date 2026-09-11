"""CLI 用 Unicode 表と公式 tokenizer の参照列を再生成する（実行時には不要）。

リポジトリの tools/.venv/bin/python で実行する。公式のローカル tokenizer を使用し、
取得・重みの変換・既存 fixture の上書きは行わない。--out の結果を Deno テストで確認してから採用する。
"""

import argparse
import json
import random
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools" / "export-recipes"))
    from anima import text as at
    from transformers import AutoTokenizer

    unicode_path = args.out / "llm-unicode.json"
    parity_path = args.out / "llm-tokenizer-parity.json"
    if unicode_path.exists() or parity_path.exists():
        raise FileExistsError(
            "出力先に生成済みの表があります。別の --out を指定してください。"
        )
    classes = at.build_char_classes()
    print("classes", at.verify_char_classes(classes), flush=True)
    fold = at.build_case_fold()
    segments = at.build_nfc_segments()
    print("nfc", at.verify_nfc_segments(segments, fuzz=2000), flush=True)
    unicode_data = {
        "tokenizers": version("tokenizers"),
        "classes": classes,
        "caseFold": sorted(fold.items()),
        "nfcSegments": segments,
    }

    prompts = [
        "The capital of France is",
        "日本の首都は",
        "WebGPUとは何ですか？日本語で一文で答えてください。",
        "A1234567B １２３４５６７ ١٢٣٤٥٦٧ 123 45",
        "e\u0301 é \u0818\u089a \u1100\u1161",
        "it'ſs can'T we're I'LL",
        "\t\n  alpha  \r\n beta\t",
        "こんにちは 👩🏽\u200d💻🙂𐐀",
        "<|im_start|>user\nhello<|im_end|>\n",
        "\ufeffhello\x00tail",
        "<unused_token_0>",
        "What is the capital of France? Answer with the city name only.",
        "日本の首都を都市名だけで答えてください。",
    ]
    rng = random.Random(111)
    alphabet = list("abcdAZ0123456789 \n\t.,!?日本語フランス") + [
        "é",
        "\u0301",
        "\u089a",
        "\u0818",
        "👩",
        "💻",
        "\u200d",
        "\u00a0",
        "\u2007",
        "\u2028",
        "\r",
        "ſ",
        "𝟘",
        "Ⅷ",
    ]
    texts = prompts + [
        "".join(rng.choices(alphabet, k=rng.randrange(1, 100))) for _ in range(60)
    ]
    fixtures = {}
    for family, folder in [
        ("qwen3", "qwen3/Qwen3-0.6B"),
        ("minicpm5", "minicpm5/MiniCPM5-2B"),
    ]:
        tok = AutoTokenizer.from_pretrained(
            repo / "inputs" / folder, local_files_only=True
        )
        cases = []
        for text in texts:
            ids = tok.encode(text, add_special_tokens=True)
            cases.append(
                {
                    "text": text,
                    "completion": " ".join(map(str, ids)),
                    "decoded": tok.decode(
                        ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                }
            )
        chats = []
        for system in [None, "Answer briefly. 日本語で答えてください。", ""]:
            for prompt in prompts[:3] + prompts[-2:]:
                messages = (
                    [] if system is None else [{"role": "system", "content": system}]
                )
                messages.append({"role": "user", "content": prompt})
                chats.append(
                    {
                        "prompt": prompt,
                        "system": system,
                        "ids": " ".join(
                            map(
                                str,
                                tok.apply_chat_template(
                                    messages,
                                    tokenize=True,
                                    return_dict=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False,
                                ),
                            )
                        ),
                    }
                )
        fixtures[family] = {"cases": cases, "chats": chats}

    args.out.mkdir(parents=True, exist_ok=True)
    unicode_path.write_text(json.dumps(unicode_data, ensure_ascii=False) + "\n")
    parity_path.write_text(json.dumps(fixtures, ensure_ascii=False) + "\n")
    print(f"生成しました: {args.out}")


if __name__ == "__main__":
    main()
