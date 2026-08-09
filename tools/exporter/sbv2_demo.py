"""examples/sbv2 デモの資産 prep と torch 参照（3 サブコマンド）。

`export_sbv2.py` / `export_deberta.py` が **グラフ**を出すのに対し、こちらが扱うのは
「テキスト → front 入力」に要るホスト側の資産と、デモ出力の**数値パリティ**だけ。
モデルグラフには一切触らない（既存の emit 経路・golden は変更しない）。

    uv run --group sbv2 python sbv2_demo.py assets
    uv run --group sbv2 python sbv2_demo.py reference \\
        --dump ../../outputs/demo/sbv2-dump/dump.safetensors
    uv run --group sbv2 python sbv2_demo.py official --text "こんにちは。"

MUST（1 プロセス 1 サブコマンド）: `patch_sbv2` のパッチはクラス属性のプロセス全域差し替えで、
`reference` はそれを当てる。`official` は**パッチ前の原経路**を走らせるのが主張の中身なので、
同一プロセスで両方を走らせると official が黙ってパッチ後の経路になる。argparse の
サブパーサは 1 プロセスにつき 1 つしか選べないため、この排他は構造的に成立する
（`export_sbv2.py` の `--verify` が対ごとの排他表を持たないのと同じ理由づけ）。

## JP-Extra の定数は「写す」のではなく「引く」

音素記号表・tone 基点・言語 ID・add_blank の挿入値は **`style_bert_vits2` の実物から
引いて JSON に落とす**（`assets` サブコマンド）。多言語版の値を手で写すと、記号表の
並びや tone 基点がずれても **shape は合ったまま音だけが壊れる**（沈黙誤値クラス）。
挿入値 0 だけは参照実装のソースにリテラルで書かれているので、
{@link blank_id_from_source} が `infer.get_text` のソースから正規表現で抜き、3 系列
（phone / tone / language）が同じ値であることまで確認する。
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

import export_sbv2
from karume import patch_sbv2
from karume.paths import OUTPUTS_ROOT

#: 実重みの置き場（`export_sbv2.py` と同じ — 綴りは向こうが持つ）。
DEFAULT_MODEL_DIR = export_sbv2.DEFAULT_MODEL_DIR
#: デモ資産の置き場。系列（IR + io）ではないので `outputs/series/` の下ではない。
#: `outputs/` は `.gitignore` 済み。
DEFAULT_DEMO_DIR = OUTPUTS_ROOT / "sbv2-demo"

#: SBV2 JP-Extra の text front が使う BERT（`export_deberta.py` の MODEL_ID と同一）。
BERT_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"

#: BERT 特徴に使う hidden_states の**末尾からの位置**。参照実装
#: `nlp/japanese/bert_feature.py` の `res["hidden_states"][-3:-2]` そのもの。
BERT_HIDDEN_FROM_END = 3

#: 資産ファイル名。`symbols.json` / `deberta-tokenizer.json` は `karume dist` が配布形へ運び、
#: `assets.safetensors` は `reference` が `--assets` で読む（Deno 側は配布形から読む）。
SYMBOLS_FILE = "symbols.json"
TOKENIZER_FILE = "deberta-tokenizer.json"
STYLE_FILE = "assets.safetensors"


# ---- ① assets: JP-Extra の規則とスタイル資産を落とす -------------------------


def blank_id_from_source() -> int:
    """add_blank の挿入値を参照実装のソースから引く（手写しの定数にしない）。

    `infer.get_text` は `commons.intersperse(phone, 0)` を phone / tone / language の
    3 系列に当てる。ここでは 3 本すべての第 2 引数を正規表現で抜き、値が 1 種類に
    揃っていることを要求する。上流でこのリテラルが変われば AssertionError で落ちる
    （黙って別の音素列になる経路を塞ぐ）。
    """
    from style_bert_vits2.models import infer as infer_module

    source = inspect.getsource(infer_module.get_text)
    found = re.findall(r"commons\.intersperse\((?:phone|tone|language),\s*(-?\d+)\)", source)
    if len(found) != 3:
        raise AssertionError(
            f"infer.get_text の intersperse 呼び出しが 3 本見つからない（{len(found)} 本）"
            " — add_blank の規則が上流で変わった可能性"
        )
    values = {int(value) for value in found}
    if len(values) != 1:
        raise AssertionError(f"intersperse の挿入値が系列ごとに違う: {sorted(values)}")
    return values.pop()


def clean_text_ranges() -> dict[str, list[list[int]]]:
    """`BasicTokenizer._clean_text` の除去 / スペース化コードポイントを閉区間へ畳む。

    参照実装は `unicodedata.category(ch).startswith("C")` などの Unicode 分類で判定する。
    TS 側に分類表を持ち込むと ICU 版差が静かな不一致になるため、判定の正本を Python に
    一本化し、全コードポイントを実評価した結果だけを資産へ焼く。
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
            # サロゲート（Cs）。参照実装が str として受け取る経路には現れないが、
            # 分類上は control なので除去側へ置く。
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
    """SBV2 が使うのと同じ経路で DeBERTa 文字トークナイザを得る。"""
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models

    return bert_models.load_tokenizer(Languages.JP, BERT_REPO)


def jp_extra_rules(hps: Any) -> dict[str, Any]:
    """JP-Extra の ID 化規則を `style_bert_vits2` の実物から引く。

    MUST: ここに literal を書かない（`blank_id_from_source` の 0 だけが例外で、それも
    ソースから抜いた値）。**多言語版の値の盲写しがこの資産を作る動機**なので、写経に
    退化させると存在意義が消える。

    NOTE: `language` は JP-Extra でも **全 0 ではない**。`infer.get_text` は
    `cleaned_text_to_sequence(..., Languages.JP)` を通すので実音素位置は
    `LANGUAGE_ID_MAP["JP"]`、add_blank の挿入位置だけが 0 になる。
    `export_sbv2.make_language` が全 0 なのは golden の合成入力としての選択で、
    推論規則ではない（合成 golden はどんな値でも成立する）。
    """
    from style_bert_vits2.constants import (
        DEFAULT_LENGTH,
        DEFAULT_NOISE,
        DEFAULT_NOISEW,
        DEFAULT_SDP_RATIO,
        DEFAULT_STYLE,
        DEFAULT_STYLE_WEIGHT,
        VERSION,
        Languages,
    )
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
        raise ValueError(f"JP-Extra 以外のモデル（version={hps.version}）はデモの対象外")
    if not hps.data.add_blank:
        raise ValueError("add_blank=False のモデルは未対応（intersperse 規則が変わる）")

    language = Languages.JP
    return {
        "source": {
            "package": "style_bert_vits2",
            "version": VERSION,
            "modelVersion": hps.version,
            "bert": BERT_REPO,
        },
        # 音素記号表。添字が enc_p.emb の行番号。
        "symbols": list(SYMBOLS),
        "pad": PAD,
        "punctuations": list(PUNCTUATIONS),
        # cleaned_text_to_sequence の 3 規則（tone は加算、language は定数）。
        "toneStart": LANGUAGE_TONE_START_MAP[language],
        "languageId": LANGUAGE_ID_MAP[language],
        "numTones": NUM_TONES,
        "numLanguages": NUM_LANGUAGES,
        # add_blank（infer.get_text）— 挿入値はソースから抜く。
        "addBlank": bool(hps.data.add_blank),
        "blankId": blank_id_from_source(),
        # 波形長の検算に使う（audio 長 = hopLength × フレーム数）。
        "samplingRate": hps.data.sampling_rate,
        "hopLength": hps.data.hop_length,
        # 実行時ノブの既定（style_bert_vits2/constants.py）。デモの CLI 既定はこれを読む。
        "defaults": {
            "sdpRatio": DEFAULT_SDP_RATIO,
            "noiseScale": DEFAULT_NOISE,
            "noiseScaleW": DEFAULT_NOISEW,
            "lengthScale": DEFAULT_LENGTH,
            "style": DEFAULT_STYLE,
            "styleWeight": DEFAULT_STYLE_WEIGHT,
        },
        # BERT 特徴の取り出し位置（末尾から数える — 層を削った variant でも同じ規則で引ける）。
        "bertHiddenFromEnd": BERT_HIDDEN_FROM_END,
    }


def resolve_style_and_speaker(
    hps: Any, style: str | None, weight: float | None, speaker: str | None
) -> tuple[str, float, str]:
    """CLI 既定を config と `constants` から解決する（CLI に literal を置かない）。"""
    from style_bert_vits2.constants import DEFAULT_STYLE, DEFAULT_STYLE_WEIGHT

    return (
        style if style is not None else DEFAULT_STYLE,
        weight if weight is not None else DEFAULT_STYLE_WEIGHT,
        speaker if speaker is not None else next(iter(hps.data.spk2id)),
    )


def style_vector(model_dir: Path, hps: Any, style: str, weight: float) -> np.ndarray:
    """`TTSModel.__get_style_vector` と同式のスタイルベクトル `[256]`。"""
    table = np.load(model_dir / export_sbv2.STYLE_FILE)
    if table.ndim != 2:
        raise ValueError(f"style_vectors.npy の形 {table.shape} が 2 次元でない")
    style2id: dict[str, int] = hps.data.style2id
    if style not in style2id:
        raise ValueError(f"スタイル {style!r} が config の style2id {sorted(style2id)} に無い")
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
    """デモの実行時資産 3 本を書く。"""
    net_g, hps = export_sbv2.load_net_g(model_dir)
    style, weight, speaker = resolve_style_and_speaker(hps, style, style_weight, speaker)
    spk2id: dict[str, int] = hps.data.spk2id
    if speaker not in spk2id:
        raise ValueError(f"話者 {speaker!r} が config の spk2id {sorted(spk2id)} に無い")
    speaker_id = spk2id[speaker]

    out_dir.mkdir(parents=True, exist_ok=True)

    rules = jp_extra_rules(hps)
    rules["style"] = {"name": style, "id": hps.data.style2id[style], "weight": weight}
    rules["speaker"] = {"name": speaker, "id": speaker_id}
    (out_dir / SYMBOLS_FILE).write_text(
        json.dumps(rules, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    tokenizer = load_bert_tokenizer()
    vocab: dict[str, int] = tokenizer.vocab
    ordered = [""] * len(vocab)
    for token, token_id in vocab.items():
        ordered[token_id] = token
    if any(token == "" for token in ordered):
        raise AssertionError("vocab の ID に穴がある（行番号 = ID の前提が崩れている）")
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
                # 行番号 0-origin = ID。Deno 側はこの文字列をそのまま食う。
                "vocabText": "\n".join(ordered),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    style_vec = style_vector(model_dir, hps, style, weight)
    g = export_sbv2.speaker_embedding(net_g, speaker_id)
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


# ---- ② reference: dump を読んで torch でチェーンを再実行 ---------------------


def wav_pcm16(samples: np.ndarray, sampling_rate: int) -> bytes:
    """f32 モノラルを 16bit PCM の WAV バイト列にする。

    MUST: 丸めは `floor(x + 0.5)`（JS の `Math.round` と同じ規則）。Python 組み込みの
    `round` は偶数丸めなので、同じ波形から 1 LSB 違う WAV が出る — 3 本の wav を
    同じ規則で書けないと「聴き比べ」が実装差の混入した比較になる。
    """
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.floor(clipped * 32767.0 + 0.5).astype(np.int16)
    data = pcm.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sampling_rate, sampling_rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    return header + data


def dump_metadata(path: Path) -> dict[str, Any]:
    """dump safetensors の `__metadata__["demo"]` を読む（JSON 1 本に畳んである）。"""
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    if "demo" not in metadata:
        raise ValueError(f"{path} に __metadata__.demo が無い（デモの dump ではない）")
    return json.loads(metadata["demo"])


def tile_bert(hidden: torch.Tensor, word2ph: list[int]) -> torch.Tensor:
    """`bert_feature.extract_bert_feature` の tile 展開（`res[i].repeat(w,1)` の連結 → 転置）。"""
    if hidden.shape[0] != len(word2ph):
        raise AssertionError(
            f"hidden のトークン数 {hidden.shape[0]} が word2ph 長 {len(word2ph)} と違う"
        )
    return torch.cat([hidden[i].repeat(n, 1) for i, n in enumerate(word2ph)], dim=0).T.contiguous()


def run_reference(dump_path: Path, model_dir: Path, assets_path: Path, out_path: Path) -> dict:
    """dump の離散入力・乱数列から torch でチェーンを再実行し、reference.wav を書く。

    実行するのは **`patch_sbv2` のモジュール群**（`Sbv2Front` / `Sbv2Voice`）と、デモと
    同じホストグルー。つまりこの突合が測るのは「同じ計算グラフを Karume が実 GPU で
    走らせた値 vs torch CPU で走らせた値」で、パッチ前の原実装との同値は
    `export_sbv2.py --verify` が別に受け持つ（層を混ぜない）。
    """
    started = time.perf_counter()
    meta = dump_metadata(dump_path)
    tensors = load_file(str(dump_path))
    assets = load_file(str(assets_path))
    g = assets["g"].to(torch.float32)
    style_vec = assets["style_vec"].to(torch.float32)

    # --- テキスト層のパリティ: 同じ bert_text から同じ input_ids が出ること -------
    # Deno 側の文字トークナイザ移植が参照実装と食い違えば、BERT 特徴が音素へ誤配置
    # されて「音は出るが崩れる」形で沈黙する。波形の突合より手前で落とす。
    tokenizer = load_bert_tokenizer()
    expected_ids = tokenizer(meta["bertText"])["input_ids"]
    dumped_ids = tensors["input_ids"].reshape(-1).tolist()
    if expected_ids != dumped_ids:
        raise AssertionError(
            "DeBERTa トークナイズが Deno 実装と食い違う"
            f"（python={expected_ids} / deno={dumped_ids}）"
        )

    # --- DeBERTa（transformers・eager）--------------------------------------
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
    hidden = hidden_states[-meta["bertHiddenFromEnd"]][0]
    word2ph = tensors["word2ph"].reshape(-1).tolist()
    bert_feature = tile_bert(hidden, word2ph).unsqueeze(0)

    # --- front（パッチ後の融合グラフ）----------------------------------------
    net_g, hps = export_sbv2.load_net_g(model_dir)
    patch_sbv2.apply_all_patches()
    front = patch_sbv2.Sbv2Front(net_g)
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

    # --- ホストグルー（デモ main.ts と同式）----------------------------------
    knobs = meta["knobs"]
    logw = logw_sdp * knobs["sdpRatio"] + logw_dp * (1.0 - knobs["sdpRatio"])
    w = torch.exp(logw) * x_mask * knobs["lengthScale"]
    w_ceil = torch.ceil(w).to(torch.int64).reshape(-1)
    dumped_w_ceil = tensors["w_ceil"].reshape(-1).to(torch.int64)
    ceil_match = bool(torch.equal(w_ceil, dumped_w_ceil))
    # 食い違った位置は値そのものを見せる（「w が整数の直上にいて GPU/CPU の 1e-5 差で
    # ceil が飛んだ」フレークか、ホストグルーの実装差かを読み手が判定できる形）。
    ceil_diffs = [
        {
            "index": int(i),
            "torch_w": float(w.reshape(-1)[i]),
            "torch_ceil": int(w_ceil[i]),
            "karume_ceil": int(dumped_w_ceil[i]),
        }
        for i in torch.nonzero(w_ceil != dumped_w_ceil).reshape(-1).tolist()
    ]
    # 以降は **dump 側の w_ceil** を使う（乱数列 zp_noise の形が Karume 側の Ty で
    # 決まっているため。torch 側の Ty で組み直すと「別の入力での比較」になる）。
    expand_idx = torch.repeat_interleave(torch.arange(dumped_w_ceil.shape[0]), dumped_w_ceil)
    total_frames = int(expand_idx.shape[0])
    zp_noise = tensors["zp_noise"].to(torch.float32)
    z_p = (
        m_p[:, :, expand_idx] + zp_noise * torch.exp(logs_p[:, :, expand_idx]) * knobs["noiseScale"]
    )
    y_mask = torch.ones(1, 1, total_frames)
    idx_k, valid = patch_sbv2.build_relattn_tables(total_frames, export_sbv2.EXPECTED_WINDOW_SIZE)

    # --- voice（flow + dec 融合）--------------------------------------------
    export_sbv2.ensure_dec_plain(net_g)
    voice = patch_sbv2.Sbv2Voice(net_g)
    with torch.no_grad():
        audio = voice(z_p, y_mask, g, idx_k, valid).reshape(-1)

    reference = audio.numpy()
    got = tensors["audio"].reshape(-1).to(torch.float32).numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(wav_pcm16(reference, meta["samplingRate"]))

    metrics: dict[str, Any] = {"length_match": reference.shape[0] == got.shape[0]}
    if metrics["length_match"]:
        diff = np.abs(reference.astype(np.float64) - got.astype(np.float64))
        # maxRel は 0 除算を避けるためだけに下限を置く（波形は零交差の連続なので、
        # rel は「その要素の |ref| が小さい」ことしか意味しない — 判定の主役は abs）。
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


# ---- ③ official: style_bert_vits2 の公式 infer（pyopenjtalk 経路）-----------


def run_official(
    model_dir: Path,
    out_path: Path,
    *,
    text: str,
    style: str | None,
    style_weight: float | None,
    speaker: str | None,
) -> dict[str, Any]:
    """公式 `infer()` で同じテキストを合成する（アクセント付与が yomi と違い得る聴き比べ用）。

    MUST: このサブコマンドは**パッチを当てない**。原実装の g2p（pyopenjtalk）・原実装の
    注意 / spline を通した音が主張の中身で、`reference` と同居させると黙ってパッチ後の
    経路になる（モジュール冒頭の 1 プロセス 1 サブコマンド）。
    """
    started = time.perf_counter()
    if patch_sbv2.patches_applied():
        raise RuntimeError("official はパッチ未適用のプロセスでのみ走らせる")

    from style_bert_vits2.constants import Languages
    from style_bert_vits2.models.infer import infer
    from style_bert_vits2.nlp import bert_models

    bert_models.load_tokenizer(Languages.JP, BERT_REPO)
    # MUST: f32 へ揃える。transformers 5.x はチェックポイントの dtype（この repo は f16）を
    # そのまま採るため、既定のままだと bert 特徴が Half で出て net_g（f32）の conv で落ちる。
    # デモ 3 本の wav を同じ精度で比べるためにも f32 に固定する。
    bert_models.load_model(Languages.JP, BERT_REPO).float()

    net_g, hps = export_sbv2.load_net_g(model_dir)
    style, weight, speaker = resolve_style_and_speaker(hps, style, style_weight, speaker)
    style_vec = style_vector(model_dir, hps, style, weight)
    knobs = jp_extra_rules(hps)["defaults"]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    # MUST: required=True。サブコマンド無しで走らせられると「1 プロセス 1 サブコマンド」の
    # 排他が「何も選ばない」で抜けられる形になる。
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("assets", "symbols.json / tokenizer / style 資産を書く"),
        ("official", "公式 infer（pyopenjtalk 経路）で合成する"),
    ):
        target = sub.add_parser(name, help=help_text)
        target.add_argument(
            "--style", default=None, help="config の style2id（既定 DEFAULT_STYLE）"
        )
        target.add_argument("--style-weight", type=float, default=None)
        target.add_argument("--speaker", default=None, help="config の spk2id（既定は先頭）")
        if name == "official":
            target.add_argument("--text", required=True)

    reference = sub.add_parser("reference", help="デモの dump から torch でチェーンを再実行する")
    reference.add_argument("--dump", type=Path, required=True)
    reference.add_argument("--assets", type=Path, default=DEFAULT_DEMO_DIR / STYLE_FILE)
    reference.add_argument("--out", type=Path, default=None, help="既定は dump と同じ場所")

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
