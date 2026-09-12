"""公開 test split の固定版を取得し、元ファイルと JSON の指紋を残す。"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq


def sources() -> tuple[dict, ...]:
    return (
        {
            "name": "arc",
            "repo": "allenai/ai2_arc",
            "revision": "210d026faf9955653af8916fad021475a3f00453",
            "file": "ARC-Easy/test-00000-of-00001.parquet",
            "licenses": ["cc-by-sa-4.0"],
            "sha256": "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177",
        },
        {
            "name": "wiki",
            "repo": "Salesforce/wikitext",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "file": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
            "licenses": ["cc-by-sa-3.0", "gfdl"],
            "sha256": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
        },
    )


def fetch(destination: Path) -> None:
    destination.mkdir()
    manifest = []
    for source in sources():
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"{source['name']}: 固定版の指紋が一致しません")
        path = destination / f"{source['name']}.parquet"
        with path.open("xb") as file:
            file.write(raw)
        rows = pq.read_table(path).to_pylist()
        text = json.dumps(rows, ensure_ascii=False).encode()
        with (destination / f"{source['name']}.json").open("xb") as file:
            file.write(text)
        card_url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/README.md"
        with urllib.request.urlopen(card_url, timeout=60) as response:
            card = response.read()
        with (destination / f"{source['name']}-dataset-card.md").open("xb") as file:
            file.write(card)
        manifest.append(
            {
                **source,
                "url": url,
                "rows": len(rows),
                "jsonSha256": hashlib.sha256(text).hexdigest(),
                "cardUrl": card_url,
                "cardSha256": hashlib.sha256(card).hexdigest(),
            }
        )
    with (destination / "sources.json").open("x") as file:
        json.dump(manifest, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    fetch(parser.parse_args().out)
