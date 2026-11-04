"""Asset prep and torch reference for the examples/sbv2 demo (3 subcommands).

Where `sbv2/export.py` / `deberta/export.py` emit **graphs**, this module handles only the
host-side assets needed for "text → front inputs" and the **numeric parity** of the demo output.
It does not touch the model graphs at all (the existing emit path and goldens are unchanged).

    uv run --group sbv2 python -m sbv2.demo assets
    uv run --group sbv2 python -m sbv2.demo reference \\
        --dump ../../outputs/examples/karume-sbv2-jvnv/sbv2-dump/dump.safetensors
    uv run --group sbv2 python -m sbv2.demo official --text "こんにちは。"

MUST (one subcommand per process): the patches of `sbv2.patch` replace class attributes
process-wide, and `reference` applies them. The whole point of `official` is to run **the original
path before patching**, so running both in one process would silently put official on the patched
path. argparse lets a process pick only one subparser, so this exclusion holds structurally
(the same reasoning as `--verify` of `sbv2/export.py` not keeping a pairwise exclusion table).

## The JP-Extra constants are "looked up", not "copied"

The phoneme symbol table, the tone base, the language ID and the add_blank insertion value are
**looked up from the real `style_bert_vits2` and written to JSON** (the `assets` subcommand). If
the values of the multilingual version were copied by hand, a shifted symbol order or tone base
would **keep the shapes matching and break only the audio** (the silent-wrong-value class).
Only the insertion value 0 is written as a literal in the reference implementation's source, so
{@link blank_id_from_source} extracts it with a regex from the source of `infer.get_text` and also
checks that the 3 sequences (phone / tone / language) use the same value.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import struct
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from _shared.paths import MISC_ROOT

from . import export, patch

#: Where the real weights live (the same as `sbv2/export.py` — that module owns the spelling).
DEFAULT_MODEL_DIR = export.DEFAULT_MODEL_DIR
#: Where the demo assets live. They are neither a series (IR + io) nor benchmark output but host
#: assets that need a re-emit when deleted, so they go under `outputs/misc/`. `outputs/` is in
#: `.gitignore`.
DEFAULT_DEMO_DIR = MISC_ROOT / "sbv2-demo"

#: The BERT the SBV2 JP-Extra text front uses (identical to MODEL_ID of `deberta/export.py`).
BERT_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"

#: The **position from the end** of hidden_states used for the BERT features. Exactly
#: `res["hidden_states"][-3:-2]` of the reference implementation's `nlp/japanese/bert_feature.py`.
#:
#: MUST: this is the position in **the full 24-layer torch model**, not the output position of the
#: distributed graph (that one is {@link BERT_GRAPH_HIDDEN_FROM_END}). The reference computation
#: uses this constant directly — taking it from the distribution's `symbols.json` would make only
#: the reference point at a different layer as soon as layers are cut.
BERT_HIDDEN_FROM_END = 3

#: The position at which **the same tensor** is taken from the distributed graph (the 22-layer
#: variant of `deberta/export.py`). The graph is truncated at the layer SBV2 uses, so it is the
#: first of the final outputs.
#:
#: That both point at the same tensor is checked at assembly time
#: (`sbv2.distribution.assert_bert_hidden` — a 3-point check of 22 layers × 1 output × position 1.
#: ADR 0045 decision 4).
BERT_GRAPH_HIDDEN_FROM_END = 1

#: Asset file names. `karume dist` carries `symbols.json` / `deberta-tokenizer.json` into the
#: distribution, and `reference` reads `assets.safetensors` via `--assets` (the Deno side reads
#: from the distribution).
SYMBOLS_FILE = "symbols.json"
TOKENIZER_FILE = "deberta-tokenizer.json"
STYLE_FILE = "assets.safetensors"


# ---- ① assets: write out the JP-Extra rules and the style assets -------------


def blank_id_from_source() -> int:
    """Takes the add_blank insertion value from the reference implementation's source (not a
    hand-copied constant).

    `infer.get_text` applies `commons.intersperse(phone, 0)` to the 3 sequences phone / tone /
    language. This extracts the second argument of all 3 with a regex and requires them to share
    one value. If upstream changes this literal, it fails with AssertionError (closing the path that
    would silently produce a different phoneme sequence).
    """
    from style_bert_vits2.models import infer as infer_module

    source = inspect.getsource(infer_module.get_text)
    found = re.findall(r"commons\.intersperse\((?:phone|tone|language),\s*(-?\d+)\)", source)
    if len(found) != 3:
        raise AssertionError(
            f"expected 3 intersperse calls in infer.get_text, found {len(found)}"
            " — the add_blank rule may have changed upstream"
        )
    values = {int(value) for value in found}
    if len(values) != 1:
        raise AssertionError(f"intersperse insertion values differ per sequence: {sorted(values)}")
    return values.pop()


def clean_text_ranges() -> dict[str, list[list[int]]]:
    """Folds the code points `BasicTokenizer._clean_text` removes / turns into spaces into closed
    ranges.

    The reference implementation decides by Unicode category (`unicodedata.category(ch)
    .startswith("C")` and the like). Bringing a category table into TS would turn ICU version
    differences into silent mismatches, so the source of truth for the decision stays in Python and
    only the result of evaluating every code point is baked into the asset.
    """

    def is_control(ch: str) -> bool:
        if ch in ("\t", "\n", "\r"):
            return False
        return unicodedata.category(ch).startswith("C")

    def is_whitespace(ch: str) -> bool:
        if ch in (" ", "\t", "\n", "\r"):
            return True
        return unicodedata.category(ch) == "Zs"

    removed: list[list[int]] = []
    spaced: list[list[int]] = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            # Surrogates (Cs). They never appear on the path where the reference implementation
            # receives a str, but by category they are control, so they go to the removed side.
            target = removed
        else:
            ch = chr(cp)
            if cp == 0 or cp == 0xFFFD or is_control(ch):
                target = removed
            elif is_whitespace(ch):
                target = spaced
            else:
                continue
        if target and target[-1][1] == cp - 1:
            target[-1][1] = cp
        else:
            target.append([cp, cp])
    return {"removed": removed, "spaced": spaced}


def load_bert_tokenizer() -> Any:
    """Gets the DeBERTa character tokenizer through the same path SBV2 uses."""
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models

    return bert_models.load_tokenizer(Languages.JP, BERT_REPO)


def jp_extra_rules(hps: Any) -> dict[str, Any]:
    """Takes the JP-Extra ID rules from the real `style_bert_vits2`.

    MUST: no literals here (the 0 of `blank_id_from_source` is the only exception, and that too is
    a value extracted from the source). **Blindly copying the multilingual values is the reason this
    asset exists**, so letting it degrade into hand-copying removes its purpose.

    NOTE: `language` is **not all 0** even for JP-Extra. `infer.get_text` goes through
    `cleaned_text_to_sequence(..., Languages.JP)`, so the real phoneme positions get
    `LANGUAGE_ID_MAP["JP"]` and only the add_blank insertion positions are 0.
    `sbv2.export.make_language` being all 0 is a choice for the synthetic golden input, not an
    inference rule (a synthetic golden holds for any value).

    MUST: only **rules shared by every model** go in. This one file is shipped to every speaker as a
    shared seat of the distribution (`shared/text/symbols.json`), so baking model-specific facts
    into it (style / speaker / the config `version`) would declare the wrong model name for other
    speakers. Runtime knob defaults are not included either — the source of truth is the manifest's
    `pipelineConfig.defaults` (`sbv2.distribution.sbv2_knob_defaults`), and the TS side does not
    read knobs from here (2026-09-24 repository-wide review W-RC5-1).
    """
    from style_bert_vits2.constants import VERSION, Languages
    from style_bert_vits2.nlp.symbols import (
        LANGUAGE_ID_MAP,
        LANGUAGE_TONE_START_MAP,
        NUM_LANGUAGES,
        NUM_TONES,
        PAD,
        PUNCTUATIONS,
        SYMBOLS,
    )

    if not hps.version.endswith("JP-Extra"):
        raise ValueError(f"a non-JP-Extra model (version={hps.version}) is out of the demo's scope")
    if not hps.data.add_blank:
        raise ValueError("add_blank=False models are not supported (the intersperse rule differs)")

    language = Languages.JP
    return {
        "source": {
            "package": "style_bert_vits2",
            "version": VERSION,
            "bert": BERT_REPO,
        },
        # Phoneme symbol table. The index is the row number of enc_p.emb.
        "symbols": list(SYMBOLS),
        "pad": PAD,
        "punctuations": list(PUNCTUATIONS),
        # The 3 rules of cleaned_text_to_sequence (tone is an offset, language is a constant).
        "toneStart": LANGUAGE_TONE_START_MAP[language],
        "languageId": LANGUAGE_ID_MAP[language],
        "numTones": NUM_TONES,
        "numLanguages": NUM_LANGUAGES,
        # add_blank (infer.get_text) — the insertion value is extracted from the source.
        "addBlank": bool(hps.data.add_blank),
        "blankId": blank_id_from_source(),
        # Used to cross-check the waveform length (audio length = hopLength × frame count).
        "samplingRate": hps.data.sampling_rate,
        "hopLength": hps.data.hop_length,
        # Where the BERT features are taken (counted from the end of **the distributed graph's
        # outputs** — a different thing from the position on the reference side).
        "bertHiddenFromEnd": BERT_GRAPH_HIDDEN_FROM_END,
        # The rule the host needs to build the relative-position index tables (from the DeBERTa
        # config — not hand-copied).
        "bertRelPos": bert_rel_pos_rule(),
    }


def bert_rel_pos_rule() -> dict[str, int]:
    """Takes the DeBERTa relative-position bucket rule from the config (the 2 values needed to build
    the tables).

    The tables themselves are graph inputs, so the host builds them for the real length
    (`rel-pos-tables.ts`). The formula's source of truth is `deberta.patch.build_rel_pos_tables`,
    and a parity test using the golden io pins the byte identity of the two.

    Falling back to `max_position_embeddings` when `max_relative_positions` is below 1 is exactly
    the rule of `DisentangledSelfAttention.__init__` (copying it would mean maintaining it twice,
    so it is resolved in this one place here too).
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(BERT_REPO)
    max_position = int(config.max_relative_positions)
    if max_position < 1:
        max_position = int(config.max_position_embeddings)
    return {"positionBuckets": int(config.position_buckets), "maxPosition": max_position}


def resolve_style_and_speaker(
    hps: Any, style: str | None, weight: float | None, speaker: str | None
) -> tuple[str, float, str]:
    """Resolves the CLI defaults from the config and `constants` (no literals in the CLI)."""
    from style_bert_vits2.constants import DEFAULT_STYLE, DEFAULT_STYLE_WEIGHT

    return (
        style if style is not None else DEFAULT_STYLE,
        weight if weight is not None else DEFAULT_STYLE_WEIGHT,
        speaker if speaker is not None else next(iter(hps.data.spk2id)),
    )


def style_vector(model_dir: Path, hps: Any, style: str, weight: float) -> np.ndarray:
    """The style vector `[256]`, computed with the same formula as `TTSModel.__get_style_vector`."""
    table = np.load(model_dir / export.STYLE_FILE)
    if table.ndim != 2:
        raise ValueError(f"the shape {table.shape} of style_vectors.npy is not 2-D")
    style2id: dict[str, int] = hps.data.style2id
    if style not in style2id:
        raise ValueError(f"style {style!r} is not in the config's style2id {sorted(style2id)}")
    mean = table[0]
    picked = table[style2id[style]]
    return np.ascontiguousarray(mean + (picked - mean) * weight, dtype=np.float32)


def emit_assets(
    model_dir: Path,
    out_dir: Path,
    *,
    style: str | None,
    style_weight: float | None,
    speaker: str | None,
) -> dict[str, Any]:
    """Writes the demo's 3 runtime assets."""
    net_g, hps = export.load_net_g(model_dir)
    style, weight, speaker = resolve_style_and_speaker(hps, style, style_weight, speaker)
    spk2id: dict[str, int] = hps.data.spk2id
    if speaker not in spk2id:
        raise ValueError(f"speaker {speaker!r} is not in the config's spk2id {sorted(spk2id)}")
    speaker_id = spk2id[speaker]

    out_dir.mkdir(parents=True, exist_ok=True)

    rules = jp_extra_rules(hps)
    (out_dir / SYMBOLS_FILE).write_text(
        json.dumps(rules, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    tokenizer = load_bert_tokenizer()
    vocab: dict[str, int] = tokenizer.vocab
    ordered = [""] * len(vocab)
    for token, token_id in vocab.items():
        ordered[token_id] = token
    if any(token == "" for token in ordered):
        raise AssertionError("the vocab IDs have gaps (the row number = ID assumption is broken)")
    (out_dir / TOKENIZER_FILE).write_text(
        json.dumps(
            {
                "source": BERT_REPO,
                "special": {
                    "clsId": tokenizer.cls_token_id,
                    "sepId": tokenizer.sep_token_id,
                    "unkId": tokenizer.unk_token_id,
                },
                "cleanRanges": clean_text_ranges(),
                # 0-based row number = ID. The Deno side consumes this string as is.
                "vocabText": "\n".join(ordered),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    style_vec = style_vector(model_dir, hps, style, weight)
    g = export.speaker_embedding(net_g, speaker_id)
    save_file(
        {"style_vec": torch.from_numpy(style_vec).reshape(1, -1), "g": g.contiguous()},
        str(out_dir / STYLE_FILE),
        metadata={
            "style": style,
            "style_weight": str(weight),
            "speaker": speaker,
            "speaker_id": str(speaker_id),
        },
    )
    return {
        "dir": str(out_dir),
        "symbols": len(rules["symbols"]),
        "toneStart": rules["toneStart"],
        "languageId": rules["languageId"],
        "blankId": rules["blankId"],
        "vocab": len(ordered),
        "style_vec": list(style_vec.shape),
        "g": list(g.shape),
    }


# ---- ② reference: read the dump and rerun the chain in torch ----------------


def wav_pcm16(samples: np.ndarray, sampling_rate: int) -> bytes:
    """Turns f32 mono into the bytes of a 16-bit PCM WAV.

    MUST: rounding is `floor(x + 0.5)` (the same rule as JS `Math.round`). Python's built-in `round`
    rounds half to even, so the same waveform would give a WAV 1 LSB off — unless the 3 wavs are
    written with the same rule, the listening comparison mixes in implementation differences.
    """
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.floor(clipped * 32767.0 + 0.5).astype(np.int16)
    data = pcm.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sampling_rate, sampling_rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    return header + data


def dump_metadata(path: Path) -> dict[str, Any]:
    """Reads `__metadata__["demo"]` of the dump safetensors (folded into one JSON string)."""
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    if "demo" not in metadata:
        raise ValueError(f"{path} has no __metadata__.demo (it is not a demo dump)")
    return json.loads(metadata["demo"])


def tile_bert(hidden: torch.Tensor, word2ph: list[int]) -> torch.Tensor:
    """The tile expansion of `bert_feature.extract_bert_feature` (concatenating `res[i].repeat(w,1)`
    → transpose)."""
    if hidden.shape[0] != len(word2ph):
        raise AssertionError(
            f"the token count {hidden.shape[0]} of hidden differs from the word2ph length"
            f" {len(word2ph)}"
        )
    return torch.cat([hidden[i].repeat(n, 1) for i, n in enumerate(word2ph)], dim=0).T.contiguous()


def run_reference(dump_path: Path, model_dir: Path, assets_path: Path, out_path: Path) -> dict:
    """Reruns the chain in torch from the dump's discrete inputs and noise, and writes
    reference.wav.

    What runs is **the modules of `sbv2.patch`** (`Sbv2Front` / `Sbv2Voice`) and the same host glue
    as the demo. So this comparison measures "the same computation graph run by Karume on a real GPU
    vs run by torch on the CPU"; equivalence with the original implementation before patching is
    handled separately by `sbv2.export --verify` (the layers are not mixed).
    """
    started = time.perf_counter()
    meta = dump_metadata(dump_path)
    tensors = load_file(str(dump_path))
    assets = load_file(str(assets_path))
    g = assets["g"].to(torch.float32)
    style_vec = assets["style_vec"].to(torch.float32)

    # --- Text-layer parity: the same bert_text gives the same input_ids ---------
    # If the port of the character tokenizer on the Deno side disagrees with the reference
    # implementation, the BERT features land on the wrong phonemes and it fails silently as
    # "there is audio, but it is garbled". Fail before the waveform comparison.
    tokenizer = load_bert_tokenizer()
    expected_ids = tokenizer(meta["bertText"])["input_ids"]
    dumped_ids = tensors["input_ids"].reshape(-1).tolist()
    if expected_ids != dumped_ids:
        raise AssertionError(
            "DeBERTa tokenization disagrees with the Deno implementation"
            f" (python={expected_ids} / deno={dumped_ids})"
        )

    # --- DeBERTa (transformers, eager) --------------------------------------

    from transformers import DebertaV2Model

    bert = DebertaV2Model.from_pretrained(
        BERT_REPO, dtype=torch.float32, attn_implementation="eager"
    )
    bert.eval()
    with torch.no_grad():
        hidden_states = bert(
            input_ids=tensors["input_ids"].to(torch.int64),
            attention_mask=tensors["attention_mask"].to(torch.int64),
            output_hidden_states=True,
        ).hidden_states
    # MUST: the reference is the **untruncated full 24-layer** model, so the position uses
    # BERT_HIDDEN_FROM_END directly (`meta["bertHiddenFromEnd"]` is the position on the distributed
    # graph's side, which is 1 for the 22-layer graph).
    hidden = hidden_states[-BERT_HIDDEN_FROM_END][0]
    word2ph = tensors["word2ph"].reshape(-1).tolist()
    bert_feature = tile_bert(hidden, word2ph).unsqueeze(0)

    # --- front (the fused graph after patching) ------------------------------
    net_g, hps = export.load_net_g(model_dir)
    patch.apply_all_patches()
    front = patch.Sbv2Front(net_g)
    x_mask = tensors["x_mask"].to(torch.float32)
    with torch.no_grad():
        logw_sdp, logw_dp, m_p, logs_p = front(
            tensors["x"].to(torch.int64),
            x_mask,
            tensors["tone"].to(torch.int64),
            tensors["language"].to(torch.int64),
            bert_feature,
            style_vec,
            g,
            tensors["z_noise"].to(torch.float32),
        )

    # --- Host glue (the same formulas as the demo's main.ts) -----------------
    knobs = meta["knobs"]
    logw = logw_sdp * knobs["sdpRatio"] + logw_dp * (1.0 - knobs["sdpRatio"])
    w = torch.exp(logw) * x_mask * knobs["lengthScale"]
    w_ceil = torch.ceil(w).to(torch.int64).reshape(-1)
    dumped_w_ceil = tensors["w_ceil"].reshape(-1).to(torch.int64)
    ceil_match = bool(torch.equal(w_ceil, dumped_w_ceil))
    # Show the values themselves at mismatched positions (so the reader can tell a flake where "w
    # sat just above an integer and a 1e-5 GPU/CPU difference flipped the ceil" from an
    # implementation difference in the host glue).
    ceil_diffs = [
        {
            "index": int(i),
            "torch_w": float(w.reshape(-1)[i]),
            "torch_ceil": int(w_ceil[i]),
            "karume_ceil": int(dumped_w_ceil[i]),
        }
        for i in torch.nonzero(w_ceil != dumped_w_ceil).reshape(-1).tolist()
    ]
    # From here on, use **the dump's w_ceil** (the shape of the noise zp_noise is fixed by the Ty of
    # the Karume side; rebuilding it with the torch side's Ty would compare different inputs).
    expand_idx = torch.repeat_interleave(torch.arange(dumped_w_ceil.shape[0]), dumped_w_ceil)
    total_frames = int(expand_idx.shape[0])
    zp_noise = tensors["zp_noise"].to(torch.float32)
    z_p = (
        m_p[:, :, expand_idx] + zp_noise * torch.exp(logs_p[:, :, expand_idx]) * knobs["noiseScale"]
    )
    y_mask = torch.ones(1, 1, total_frames)
    idx_k, valid = patch.build_relattn_tables(total_frames, export.EXPECTED_WINDOW_SIZE)

    # --- voice (flow + dec fused) -------------------------------------------
    export.ensure_dec_plain(net_g)
    voice = patch.Sbv2Voice(net_g)
    with torch.no_grad():
        audio = voice(z_p, y_mask, g, idx_k, valid).reshape(-1)

    reference = audio.numpy()
    got = tensors["audio"].reshape(-1).to(torch.float32).numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(wav_pcm16(reference, meta["samplingRate"]))

    metrics: dict[str, Any] = {"length_match": reference.shape[0] == got.shape[0]}
    if metrics["length_match"]:
        diff = np.abs(reference.astype(np.float64) - got.astype(np.float64))
        # maxRel has a floor only to avoid division by zero (a waveform is a run of zero
        # crossings, so rel only says "|ref| of that element is small" — abs is what decides).
        metrics["maxAbs"] = float(diff.max())
        metrics["maxRel"] = float((diff / np.maximum(np.abs(reference), 1e-12)).max())
        metrics["rmse"] = float(np.sqrt((diff**2).mean()))
        metrics["refMaxAbs"] = float(np.abs(reference).max())

    return {
        "dump": str(dump_path),
        "wav": str(out_path),
        "text": meta["text"],
        "version": hps.version,
        "phonemes": int(tensors["x"].shape[1]),
        "tokens": len(dumped_ids),
        "frames": total_frames,
        "samples": int(reference.shape[0]),
        "seconds": round(reference.shape[0] / meta["samplingRate"], 3),
        "w_ceil_exact": ceil_match,
        "w_ceil_diffs": ceil_diffs,
        **metrics,
        "elapsed": round(time.perf_counter() - started, 1),
    }


# ---- ③ official: the official infer of style_bert_vits2 (pyopenjtalk path) --


def run_official(
    model_dir: Path,
    out_path: Path,
    *,
    text: str,
    style: str | None,
    style_weight: float | None,
    speaker: str | None,
) -> dict[str, Any]:
    """Synthesizes the same text with the official `infer()` (for listening comparisons, since its
    accent assignment can differ from yomi's).

    MUST: this subcommand **applies no patches**. The audio through the original g2p (pyopenjtalk)
    and the original attention / spline is the whole point; sharing a process with `reference`
    would silently put it on the patched path (one subcommand per process, see the module top).
    """
    started = time.perf_counter()
    if patch.patches_applied():
        raise RuntimeError("official runs only in a process where no patches have been applied")

    from style_bert_vits2.constants import (
        DEFAULT_LENGTH,
        DEFAULT_NOISE,
        DEFAULT_NOISEW,
        DEFAULT_SDP_RATIO,
        Languages,
    )
    from style_bert_vits2.models.infer import infer
    from style_bert_vits2.nlp import bert_models

    bert_models.load_tokenizer(Languages.JP, BERT_REPO)
    # MUST: cast to f32. transformers 5.x takes the checkpoint's dtype (f16 for this repo) as is, so
    # by default the bert features come out as Half and fail in the conv of net_g (f32). It is also
    # fixed to f32 so that the demo's 3 wavs are compared at the same precision.
    bert_models.load_model(Languages.JP, BERT_REPO).float()

    net_g, hps = export.load_net_g(model_dir)
    style, weight, speaker = resolve_style_and_speaker(hps, style, style_weight, speaker)
    style_vec = style_vector(model_dir, hps, style, weight)
    # The runtime knob defaults are read directly from `style_bert_vits2.constants` (they are not
    # put into `symbols.json` — the MUST of {@link jp_extra_rules}).
    knobs = {
        "sdpRatio": DEFAULT_SDP_RATIO,
        "noiseScale": DEFAULT_NOISE,
        "noiseScaleW": DEFAULT_NOISEW,
        "lengthScale": DEFAULT_LENGTH,
    }
    audio = infer(
        text=text,
        style_vec=style_vec,
        sdp_ratio=knobs["sdpRatio"],
        noise_scale=knobs["noiseScale"],
        noise_scale_w=knobs["noiseScaleW"],
        length_scale=knobs["lengthScale"],
        sid=hps.data.spk2id[speaker],
        language=Languages.JP,
        hps=hps,
        net_g=net_g,
        device="cpu",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(wav_pcm16(audio, hps.data.sampling_rate))
    return {
        "wav": str(out_path),
        "text": text,
        "style": style,
        "speaker": speaker,
        "samples": int(audio.shape[0]),
        "seconds": round(audio.shape[0] / hps.data.sampling_rate, 3),
        "maxAbs": float(np.abs(audio).max()),
        "knobs": knobs,
        "elapsed": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    # MUST: required=True. If it could run without a subcommand, the "one subcommand per process"
    # exclusion could be bypassed by choosing nothing.
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("assets", "write the symbols.json / tokenizer / style assets"),
        ("official", "synthesize with the official infer (pyopenjtalk path)"),
    ):
        target = sub.add_parser(name, help=help_text)
        target.add_argument(
            "--style", default=None, help="a key of the config's style2id (default DEFAULT_STYLE)"
        )
        target.add_argument("--style-weight", type=float, default=None)
        target.add_argument(
            "--speaker", default=None, help="a key of the config's spk2id (default: the first)"
        )
        if name == "official":
            target.add_argument("--text", required=True)

    reference = sub.add_parser("reference", help="rerun the chain in torch from the demo's dump")
    reference.add_argument("--dump", type=Path, required=True)
    reference.add_argument("--assets", type=Path, default=DEFAULT_DEMO_DIR / STYLE_FILE)
    reference.add_argument("--out", type=Path, default=None, help="default: next to the dump")

    args = parser.parse_args()
    if args.command == "assets":
        report = emit_assets(
            args.model_dir,
            DEFAULT_DEMO_DIR,
            style=args.style,
            style_weight=args.style_weight,
            speaker=args.speaker,
        )
    elif args.command == "reference":
        report = run_reference(
            args.dump,
            args.model_dir,
            args.assets,
            args.out if args.out is not None else args.dump.parent / "reference.wav",
        )
    else:
        report = run_official(
            args.model_dir,
            DEFAULT_DEMO_DIR / "out" / "official.wav",
            text=args.text,
            style=args.style,
            style_weight=args.style_weight,
            speaker=args.speaker,
        )
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
