"""Wan2.1 のテキスト埋め込み資産（第 1 段 = 事前計算した埋め込み — ADR 0118 決定 4・段 6）。

固定プロンプト（正 3 本 + negative 1 本）を umT5-XXL に通した出力 `[L_valid, 4096]` を f32 で
safetensors に書く。DiT の入力 `[1, 512, 4096]`（有効長の後ろをゼロで埋めた形）はホストが作る —
資産は有効長ぶんだけを持つ（`L_valid` = トークナイザのマスクが 1 の長さ）。

## 作り方（上流の経路をそのまま呼ぶ）

- 埋め込みは diffusers 0.39.0 の `WanPipeline._get_t5_prompt_embeds` をそのまま呼んで作る
  （`prompt_clean` → トークナイザ `padding="max_length"`・`max_length=512` → `UMT5EncoderModel` →
  有効長で切ってゼロで 512 まで埋める）。ここは返った `[1, 512, 4096]` を有効長で切るだけで、
  式を写さない。
- umT5 は**上流と同じ dtype（bf16）**で読む（公式 `t5_dtype` = bfloat16・Diffusers ドキュメントの
  例も `torch_dtype=torch.bfloat16`）。出力の bf16 → f32 は値を変えない（bf16 は f32 の上位
  16 ビット）。
- **別プロセス・CPU**。DiT / VAE は読まない（ホスト RAM 31 GiB に umT5 と DiT を同居させない —
  決定 4）。重みは fp32 の safetensors 5 分割（約 22.7 GB）を bf16 で読む（約 11.4 GB）。
  transformers 5 の `from_pretrained` は常に省メモリの読み込みで、`low_cpu_mem_usage` は無視される
  （渡さない）。
- 正規化は `prompt_clean`（ftfy が入っているときだけ `ftfy.fix_text` — `pipeline_wan.py:78-81`）。
  ftfy は wan の依存群で `==6.3.1` に固定（決定 4 の「固定」）。正規化前の原文と正規化後の文字列を
  両方メタに書く。

## 資産の形式（`text_embeds.safetensors`）

- テンソルはプロンプトごとに 1 本（名前 = {@link FixedPrompt.name}・`F32`・`[L_valid, 4096]`）。
- メタは**キー 1 つ**（{@link METADATA_KEY}）に JSON（キー整列・区切りの空白なし）を入れる。
  MUST: キーを複数にしない — safetensors（Rust）はメタを HashMap で書き、プロセスごとにキーの
  並びが変わる（2026-10-02 実測: 同じ入力の 3 回の書き出しで sha256 が 2 通り）。キー 1 つなら
  同じ入力で同じバイトになる。
- 置き場は系列の席 `outputs/series/wan2.1-t2v-1.3b-text-embeds/`（グラフを持たない compile
  生成物 — docs/assets-layout.md）。配布形ではモデル単位の `assets`（quant 非依存 — 決定 4）。

## 固定プロンプト（利用者裁定 2026-10-02: 公式 README / Diffusers ドキュメントの例文から取る）

- `boxing-cats`（positive）: 公式 README の t2v-1.3B の例（`generate.py` の `EXAMPLE_PROMPT` と
  同文）。
- `ferret`（positive）: Diffusers ドキュメントの Wan の T2V の例（複数行の文字列のまま）。
- `cat-dog-baking`（positive）: Diffusers の `WanPipeline.__call__` の例（API リファレンスに出る
  docstring）。
- `negative`（negative）: 公式 `wan/configs/shared_config.py` の `sample_neg_prompt`（中国語）。

動きの量: 格闘（大）・カメラの突進と跳躍（大・カメラ移動）・台所の作業（小）。

    uv run --group wan --inexact python -m wan.text_embeds --fetch   # umT5 とトークナイザを取得
    uv run --group wan --inexact python -m wan.text_embeds           # 資産を書く

MUST: diffusers / transformers / ftfy は関数の中で import する（`wan` グループは既定の sync に
入らない — `tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from _shared.paths import SERIES_ROOT
from wan.pipeline_ref import MAX_SEQUENCE_LENGTH, TEXT_DIM
from wan.sources import DEFAULT_MODEL, SOURCES, text_snapshot

#: 系列の席（決定 4・決定 7）。
SERIES_NAME = "wan2.1-t2v-1.3b-text-embeds"

#: 資産のファイル名。
ASSET_NAME = "text_embeds.safetensors"

#: メタの唯一のキー（値は JSON — モジュール doc の MUST）。
METADATA_KEY = "karume.wan.text_embeds"

#: umT5 の dtype（上流と同じ — 公式 `t5_dtype`）。
TEXT_ENCODER_DTYPE = torch.bfloat16

#: 役割の語彙。
POSITIVE = "positive"
NEGATIVE = "negative"

#: 出所の commit（URL の版を固定するため — ブランチ名の URL は先頭が動くと別の文面を指しうる）。
#: 公式リポは 2026-10-02 時点の main の先頭（2026-03-05 の commit）、Diffusers は v0.39.0 の
#: タグが指す commit。
_WAN_COMMIT = "9737cba9c1c3c4d04b33fcad41c111989865d315"
_DIFFUSERS_COMMIT = "a3608b512ed7248499a44c61d954965ed9bdae4d"


@dataclass(frozen=True)
class FixedPrompt:
    """固定プロンプト 1 本（資産のテンソル名・役割・原文・出所）。"""

    name: str
    role: str
    #: 出所の原文（改行や全角の約物を含めて逐語 — 正規化は生成時に上流の関数で掛ける）。
    text: str
    #: 出所の URL（commit を固定した版）。
    url: str
    #: URL の中の位置（読み手が原文を探せる粒度）。
    locator: str


FIXED_PROMPTS: tuple[FixedPrompt, ...] = (
    FixedPrompt(
        name="boxing-cats",
        role=POSITIVE,
        text=(
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a"
            " spotlighted stage."
        ),
        url=f"https://github.com/Wan-Video/Wan2.1/blob/{_WAN_COMMIT}/README.md",
        locator=(
            "README の t2v-1.3B の実行例の --prompt"
            "（generate.py の EXAMPLE_PROMPT['t2v-1.3B'] と同文）"
        ),
    ),
    FixedPrompt(
        name="ferret",
        role=POSITIVE,
        text=(
            "\nThe camera rushes from far to near in a low-angle shot,\n"
            "revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the"
            " camera zooms in\n"
            "for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves"
            " blanket the ground.\n"
            "Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side"
            " lighting casts dynamic\n"
            "shadows and warm highlights. Medium composition, front view, low angle, with depth of"
            " field.\n"
        ),
        url=(
            f"https://github.com/huggingface/diffusers/blob/{_DIFFUSERS_COMMIT}"
            "/docs/source/en/api/pipelines/wan.md"
        ),
        locator='Text-to-Video の例（T2V memory / T2V inference speed のタブ）の prompt = """…"""',
    ),
    FixedPrompt(
        name="cat-dog-baking",
        role=POSITIVE,
        text=(
            "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring"
            " flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy,"
            " with sunlight streaming through the window."
        ),
        url=(
            f"https://github.com/huggingface/diffusers/blob/{_DIFFUSERS_COMMIT}"
            "/src/diffusers/pipelines/wan/pipeline_wan.py"
        ),
        locator=(
            "EXAMPLE_DOC_STRING の prompt（WanPipeline.__call__ の例 — ドキュメントの API"
            " リファレンスに出る）"
        ),
    ),
    FixedPrompt(
        name="negative",
        role=NEGATIVE,
        text=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
            "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
            "三条腿，背景人很多，倒着走"
        ),
        url=f"https://github.com/Wan-Video/Wan2.1/blob/{_WAN_COMMIT}/wan/configs/shared_config.py",
        locator="wan_shared_cfg.sample_neg_prompt",
    ),
)


class TextEmbedsError(ValueError):
    """埋め込み資産の形・メタが決定 4 の取り決めから外れた。"""


def prompt_by_name(name: str) -> FixedPrompt:
    """名前で固定プロンプトを引く（無ければ fail loudly）。"""
    for prompt in FIXED_PROMPTS:
        if prompt.name == name:
            return prompt
    raise TextEmbedsError(f"固定プロンプト {name!r} は無い（{[p.name for p in FIXED_PROMPTS]}）")


def normalize(text: str) -> str:
    """上流の `prompt_clean`（ftfy が入っていれば `fix_text` → html の unescape → 空白の畳み込み）。

    MUST: ftfy が import できない環境では止める — 上流は ftfy が無いと黙って `fix_text` を飛ばし、
    同じプロンプトが別の文字列（別の埋め込み）になる（決定 4）。
    """
    import ftfy  # noqa: F401 — 有無の検査（上流は `is_ftfy_available()` で分岐する）
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from diffusers.utils import is_ftfy_available

    if not is_ftfy_available():
        raise TextEmbedsError("diffusers が ftfy を見つけない — 正規化が上流と変わる")
    return prompt_clean(text)


def load_text_encoder(model: str = DEFAULT_MODEL) -> tuple[Any, Any]:
    """pin した revision のトークナイザと umT5（bf16・CPU・eval）を読む。"""
    from transformers import AutoTokenizer, UMT5EncoderModel

    snapshot = text_snapshot(model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        snapshot, subfolder="text_encoder", dtype=TEXT_ENCODER_DTYPE
    ).eval()
    if encoder.dtype != TEXT_ENCODER_DTYPE:
        raise TextEmbedsError(f"umT5 が {encoder.dtype} で読まれた（{TEXT_ENCODER_DTYPE} を指定）")
    return tokenizer, encoder


def valid_length(tokenizer: Any, normalized: str) -> int:
    """上流と同じ引数でトークナイズしたマスクの長さ（= `L_valid`）。

    MUST: 512 を超えて切り詰められたプロンプトは拒む — 上流は黙って切るが、固定プロンプトが切れて
    いれば原文と埋め込みが対応しない。
    """
    inputs = tokenizer(
        [normalized],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    length = int(inputs.attention_mask.gt(0).sum())
    unpadded = tokenizer([normalized], add_special_tokens=True).input_ids[0]
    if len(unpadded) > MAX_SEQUENCE_LENGTH:
        raise TextEmbedsError(f"トークン数 {len(unpadded)} が {MAX_SEQUENCE_LENGTH} を超える")
    if length != len(unpadded):
        raise TextEmbedsError(f"マスクの長さ {length} がトークン数 {len(unpadded)} と違う")
    return length


def encode(tokenizer: Any, encoder: Any, text: str) -> torch.Tensor:
    """上流 `_get_t5_prompt_embeds` で `[1, 512, 4096]` を作り、有効長 `[L_valid, 4096]` f32 で
    返す。

    有効長の後ろが厳密にゼロであること（上流が `new_zeros` で埋めた行）を確かめる — 有効長の
    取り違えはここで落ちる。
    """
    from diffusers import WanPipeline

    pipeline = WanPipeline(
        tokenizer=tokenizer, text_encoder=encoder, vae=None, scheduler=None, transformer=None
    )
    with torch.no_grad():
        padded = pipeline._get_t5_prompt_embeds(
            prompt=text,
            num_videos_per_prompt=1,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    if padded.shape != (1, MAX_SEQUENCE_LENGTH, TEXT_DIM) or padded.dtype != torch.float32:
        raise TextEmbedsError(f"上流の埋め込みが想定外の形 {tuple(padded.shape)} / {padded.dtype}")
    length = valid_length(tokenizer, normalize(text))
    if torch.count_nonzero(padded[0, length:]) != 0:
        raise TextEmbedsError(f"有効長 {length} の後ろにゼロでない行がある")
    if torch.count_nonzero(padded[0, length - 1]) == 0:
        raise TextEmbedsError(f"有効長 {length} の最後の行がゼロ — 有効長の取り違え")
    embeds = padded[0, :length].contiguous()
    if not bool(embeds.isfinite().all()):
        raise TextEmbedsError("埋め込みに非有限値がある")
    return embeds


def _versions() -> dict[str, str]:
    import diffusers
    import ftfy
    import transformers

    return {
        "diffusers": diffusers.__version__,
        "ftfy": ftfy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def asset_metadata(
    prompts: Sequence[FixedPrompt],
    lengths: Mapping[str, int],
    normalized: Mapping[str, str],
    *,
    model: str,
    versions: Mapping[str, str],
) -> dict[str, Any]:
    """資産のメタ（JSON に落とす前の形）。"""
    source = SOURCES[model]
    return {
        "source": {"repo": source.repo, "revision": source.revision},
        "text_encoder": {
            "class": "UMT5EncoderModel",
            "dtype": str(TEXT_ENCODER_DTYPE).removeprefix("torch."),
            "device": "cpu",
            "boundary": "last_hidden_state before the DiT text projection (valid rows only)",
        },
        "tokenizer": {"max_length": MAX_SEQUENCE_LENGTH, "padding": "max_length"},
        "normalizer": "diffusers.pipelines.wan.pipeline_wan.prompt_clean (ftfy.fix_text applied)",
        "versions": dict(versions),
        "prompts": [
            {
                "name": prompt.name,
                "role": prompt.role,
                "prompt": prompt.text,
                "normalized": normalized[prompt.name],
                "tokens": lengths[prompt.name],
                "source": {"url": prompt.url, "locator": prompt.locator},
            }
            for prompt in prompts
        ],
    }


def write_asset(
    path: Path, embeds: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> None:
    """資産を書く（staging → 置換）。形・dtype・メタの対応を書く前に確かめる。"""
    entries = metadata["prompts"]
    names = [entry["name"] for entry in entries]
    if sorted(names) != sorted(embeds) or len(set(names)) != len(names):
        raise TextEmbedsError(f"メタのプロンプト {names} とテンソル {sorted(embeds)} が対応しない")
    for entry in entries:
        tensor = embeds[entry["name"]]
        if tensor.dtype != torch.float32:
            raise TextEmbedsError(f"{entry['name']}: f32 で書く（{tensor.dtype}）")
        if tensor.dim() != 2 or tensor.shape[1] != TEXT_DIM:
            raise TextEmbedsError(
                f"{entry['name']}: [L_valid, {TEXT_DIM}]（{tuple(tensor.shape)}）"
            )
        if tensor.shape[0] != entry["tokens"] or not 0 < tensor.shape[0] <= MAX_SEQUENCE_LENGTH:
            raise TextEmbedsError(
                f"{entry['name']}: 行数 {tensor.shape[0]} が"
                f"メタのトークン数 {entry['tokens']} と違う"
            )
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    save_file(
        {name: tensor.contiguous() for name, tensor in embeds.items()},
        str(staging),
        metadata={METADATA_KEY: payload},
    )
    staging.replace(path)


def read_asset(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """資産を読む（テンソルとメタ）。

    メタのキーが {@link METADATA_KEY} 1 つでなければ fail loudly。
    """
    with safe_open(str(path), framework="pt") as handle:
        raw = handle.metadata() or {}
        if set(raw) != {METADATA_KEY}:
            raise TextEmbedsError(f"{path}: メタのキー {sorted(raw)} が {METADATA_KEY} 1 つでない")
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
    return tensors, json.loads(raw[METADATA_KEY])


def _peak_rss_gib() -> float:
    """このプロセスの RSS の最大（GiB — Linux の `ru_maxrss` は KiB）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def generate(model: str, out: Path) -> dict[str, Any]:
    """固定プロンプト全部を埋め込んで資産を書き、要約を返す。"""
    started = time.perf_counter()
    tokenizer, encoder = load_text_encoder(model)
    loaded = time.perf_counter()
    print(f"[load] umT5 {TEXT_ENCODER_DTYPE}: {loaded - started:.1f} s", flush=True)
    embeds: dict[str, torch.Tensor] = {}
    normalized: dict[str, str] = {}
    lengths: dict[str, int] = {}
    seconds: dict[str, float] = {}
    for prompt in FIXED_PROMPTS:
        began = time.perf_counter()
        embeds[prompt.name] = encode(tokenizer, encoder, prompt.text)
        normalized[prompt.name] = normalize(prompt.text)
        lengths[prompt.name] = int(embeds[prompt.name].shape[0])
        seconds[prompt.name] = round(time.perf_counter() - began, 1)
        print(f"[encode] {prompt.name}: L_valid {lengths[prompt.name]} {seconds[prompt.name]} s")
    metadata = asset_metadata(FIXED_PROMPTS, lengths, normalized, model=model, versions=_versions())
    path = out / ASSET_NAME
    write_asset(path, embeds, metadata)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "load_seconds": round(loaded - started, 1),
        "encode_seconds": seconds,
        "total_seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
        "torch_threads": torch.get_num_threads(),
        "prompts": [
            {
                "name": entry["name"],
                "role": entry["role"],
                "tokens": entry["tokens"],
                "normalized": entry["normalized"],
                "abs_max": float(embeds[entry["name"]].abs().max()),
            }
            for entry in metadata["prompts"]
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument("--out", type=Path, default=SERIES_ROOT / SERIES_NAME)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="umT5 とトークナイザを pin した revision で取得するだけ",
    )
    args = parser.parse_args(argv)
    if args.fetch:
        print(f"snapshot: {text_snapshot(args.model, fetch=True)}")
        return 0
    print(json.dumps(generate(args.model, args.out), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
