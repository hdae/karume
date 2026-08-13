"""実重み Irodori-TTS v4-Small を IR v1 コンテナ + golden io へ書き出す台本。

今回のスコープは**テキスト条件エンコーダ**（recon の G1 / G1a / G1b）・**speaker encoder /
duration predictor**（同 G2 / G3）・**DiT 1 step**（同 G5' = G4 を畳んだ形・ADR 0047）で、
codec（G6 / G7）は後続の波でこの台本にターゲットとして足す。

    cd tools/exporter
    uv run --with 'transformers==5.14.1' python export_irodori.py
    uv run --with 'transformers==5.14.1' python export_irodori.py --target backbone
    uv run --with 'transformers==5.14.1' python export_irodori.py --dtype f16
    uv run --with 'transformers==5.14.1' python export_irodori.py --dtype i8

transformers は **5.14.1 でピン**する（`export_embeddinggemma.py` と同じ理由 — モデリング
コードが変わるとグラフ形が変わる。加えて `karume.patch_irodori` が
`ModernBertAttention.forward` をクラス属性ごと差し替える）。pyproject.toml / uv.lock には
入れず `--with` で一時的に足す。

モデル実装は GitHub `Aratako/Irodori-TTS` の clone（既定 `inputs/irodori/Irodori-TTS/`）から
**`sys.path` 追加で import** する（`--source-dir`）。`irodori_tts.model` の import は
`irodori_tts/__init__.py` 経由になるが、そこが引く追加依存は transformers だけなので、
限定 import の細工は要らない。

## 何をグラフに載せるか（6 ターゲット・B=1・T / S は記号次元）

| ターゲット     | 入力          | 出力          | 中身                                  |
| -------------- | ------------- | ------------- | ------------------------------------- |
| `backbone`     | `[1,T]` ids   | `[1,T,768]`   | ModernBERT-ja-310m（25 層・共有）     |
| `text-proj`    | `[1,T,768]`   | `[1,T,512]`   | text 側 projector（residual_mlp）     |
| `caption-proj` | `[1,T,768]`   | `[1,T,512]`×2 | caption 側 projector（+ `caption_norm`）|
| `speaker`      | `[1,S,128]`   | `[1,S,768]`   | `ReferenceLatentEncoder` + 出力 norm  |
| `duration`     | 下記 5 本     | `[1]`         | `text_norm` + duration（token-sum 形）|
| `dit`          | 下記 6 本     | `[1,S,32]`    | DiT 1 step（12 層・G4 畳み込み形）    |

`duration` の入力 5 本は `text_state [1,T,512]` / `speaker_vec [1,768]` /
`has_speaker [1,1]`（bool）/ `caption_vec [1,512]` / `has_caption [1,1]`（bool）。

backbone を projector と融合しないのは、**backbone が text と caption で共有**だから
（融合すると 1.2GB の重みが 2 系列に複製される）。ホストは backbone を 2 回回して、
それぞれの projector へ流す。

### `caption-proj`（G1b）の第 2 出力

`duration`（G3）が食う `caption_vec` の契約は「**`caption_norm` を掛けた** caption 系列の
masked mean」（`DurationPredictor._caption_vec` — pooling は `masked_mean`）だが、
`caption_norm` の重みは `dit` グラフの**内側**にしか無い（`text_norm` / `caption_norm` は
消費側が内包する — ADR 0010 の理由）。ホストが masked mean を採るには norm 済みの系列が要る。

そこで `caption-proj` に**第 2 出力 = `caption_norm`（RMSNorm 512）適用済み系列**を足し、
masked mean だけをホストに残す。第 1 出力（生の projector 出力）は**そのまま**で、
`dit` の `caption_state` 入力と `duration` の鎖の両方が今までどおりこちらを食う
（`duration` の契約と golden は不変 — 再 export 後の第 1 出力が旧 golden と**ビット一致**する
ことが、この主張の実測になる）。

代替案（却下）: ① `caption_norm` の weight をホストへ配る → 学習済みの値がモデルファイルの
外に出て、正規経路がファイル 1 個で閉じなくなる（ADR 0010 の代替案 2 と同じ理由）
② masked mean までグラフに載せる → mask が実行時入力になり、`caption-proj` だけ
実行時マスクを持つターゲットになる（他 2 本のテキスト系と方式が割れる）。
`text-proj` は `text_norm` を要らない（`duration` がグラフ内で掛ける）ので**変更しない**。

### `speaker`（G2）の境界

`ReferenceLatentEncoder` の直後に来る `speaker_norm`（RMSNorm 768）まで**載せる** —
`encode_conditions` はこの 2 つを必ず続けて掛けるので、切ると RMSNorm がホスト側の
モデル計算の写しになる。逆に、その次の `_prepend_masked_mean_token`（時間平均トークンの
前置）は**載せられない**: `cat` の対象軸が記号次元 S になり、IR の `cat` は静的軸しか
受けない（実測 — `aten.cat.default: 対象軸は静的でなければならない`）。したがって

- 平均トークンの生成と前置は**ホスト**（`[1,S,768]` の軸 1 平均 + concat — 純粋な配列操作）
- `duration` が要る `speaker_state[:,0]` は、その平均トークンそのもの

MUST: 参照 latent の**マスクはグラフに持たない**。実装（`_load_reference_latent`）は
参照ありなら常に全 1 マスクを作るので、B=1 では「詰めた列で呼ぶ」ことと同値
（`_check_wrapper_equivalence` が実重みで atol 0 を実測する）。参照なし（`no_ref`）は
マスク**全 0** の別経路だが、torch 側の出力は**厳密に 0**（SDPA の safe-softmax が全マスク行に
0 を返し、末尾の `x * mask_f` が全体を 0 にする — `_no_reference_evidence` が実測）。
ホストはゼロ行列を作れば良く、グラフを呼ぶ必要が無い。

### `duration`（G3）の境界

recon の「系列入力なし」は**この重みでは成り立たない**: `duration_architecture` は
`token_sum_dual_adarn_zero_no_aux` で、text 系列を token ごとに走らせて
`log1p(Σ softplus(logit))` を返す（`model.py` の `DurationPredictor.forward`）。よって
系列入力 `[1,T,512]` が要る。載せる範囲は

- 先頭の `text_norm`（RMSNorm 512）— `text-proj` の出力から**直に鎖にする**ため
  （既存ターゲットは不変なので、norm は消費側が持つ）
- `speaker_vec` / `caption_vec` は**ホスト供給**（前者は speaker 平均トークンの切り出し、
  後者は `caption_norm` を掛けた caption 系列の masked mean）。caption 系列を graph へ入れると
  記号次元が 2 本になる（T と caption 長）ので採らない
- **`has_speaker` / `has_caption` は bool 入力**にして `null_speaker` / `null_caption` の
  選択（`torch.where`）を**グラフ内に残す**。外に出すと 2 本の学習済みベクトルが
  「モデルファイルから取り出してホストへ配る値」になり、参照なし / caption なしの正規経路が
  モデルファイル 1 個で閉じなくなる（ADR 0010 の代替案 2 を却下した理由と同じ）

### `dit`（G5' = G4 を畳んだ形）の境界

設計の正本は ADR 0047。入力 6 本は

| 名前            | shape                | 中身                                                    |
| --------------- | -------------------- | ------------------------------------------------------- |
| `x_t`           | `[1,S,32]`           | patch 済み latent（S = 記号次元・min 2 / max 750）      |
| `t_embed`       | `[1,512]`            | `get_timestep_embedding(t, 512)`（**ホスト昇格**）      |
| `mask`          | `[1,1,1,S+1519]`     | bool。self / text / speaker / caption の順に連結        |
| `text_state`    | `[1,256,512]`        | `text-proj` の出力を `max_text_len` へ右 pad            |
| `speaker_state` | `[1,751,768]`        | `speaker` の出力 + 平均トークン前置を 750+1 へ右 pad    |
| `caption_state` | `[1,512,512]`        | `caption-proj` の出力を `max_caption_len` へ右 pad      |

MUST: **`t_embed` はグラフに入れない** — `get_timestep_embedding` は `cos` を使い、`cos` は
IR の op 語彙に無い（`sin` だけを足した第 1 層の判断 — ADR 0043）。ホストが 3 行で作る。

MUST: **`text_norm` / `caption_norm` はこのグラフが内包する**（入力は projector の生の出力）。
`duration` が `text_norm` を内包しているのと同じ理由で、外に出すと 2 本の学習済み RMSNorm
weight が「モデルファイルから取り出してホストへ配る値」になり、正規経路がモデルファイル
1 個で閉じなくなる（ADR 0010 の代替案 2 を却下した理由）。pad 行は 0 で、`rms_norm(0)` は
厳密に 0 なので、pad の位置は norm を通しても 0 のまま（かつマスクで落ちる）。
speaker 側だけ norm が上流（`speaker` ターゲット）にあるのは、平均トークンの前置が
ホストに残るため（G2 の境界 — 上の節）。

**G4（context-KV 事前射影）は畳む**（ADR 0047 決定 3）: 各層の `project_context_kv` を
グラフ内で毎回計算する。別グラフにすると出力 178MB を毎 run アップロードすることになり、
再計算 59.6 GFLOP/forward を払うほうが差し引き速い。分岐点は「入力値の Session 常駐」。

**uncond はマスクだけで表す**（同決定 1）: 上流の CFG は text / speaker / caption の
各 uncond で「state を 0 にした context KV」をもう一組作るが、マスクが 0 の区間の寄与は
`exp(−inf)=0` で厳密に 0 なので、**cond の KV のままマスクだけ 0 にした結果とビット一致**する。
golden の uncond 3 変種は上流の uncond（state 0 + マスク 0）で参照値を採り、グラフには
**cond の state + 区間 0 のマスク**を渡す — `EAGER_EQUIV_ATOL = 0` の同値検証が通ることが、
そのままこの決定の実証になっている。

MUST: SDPA は**保存しない**（既定の分解表 = `PRESERVED_OP_PREFIXES`）。マスクが実行時の
bool 入力なので融合 attention の契約（マスク無し / 加算型 `[1,1,M,N]`）に載らず、分解経路 +
`safe_softmax`（ADR 0044）で通す。ターゲット別に `preserved` を持つのはこのため。

`aux_features`（14 次元）は token-sum 形では**一切読まれない**ので入力に持たない。
恒真化しないよう `_duration_aux_is_inert` が「別の aux を渡しても出力が 1 ビットも変わらない」
ことを毎回実測する。`_safe_attention_mask`（`model.py:429`）は全 1 マスクでは恒等
（`has_any.all()` が真で入力をそのまま返す）なので、B=1 + 「テキストは必ず 1 token 以上」の
契約下でグラフに現れない。

## 実行時 attention_mask を持たない（EmbeddingGemma と同じ静的方式）

MUST: backbone は **`attention_mask` を実行時入力として持たない**。Irodori の推論は
`max_text_len` / `max_caption_len` へ右詰め pad した固定長で backbone を呼び、pad 位置を
マスクで落とす形だが、**B=1 では「詰めきった長さ T で呼ぶ」ことと数学的に同値**
（双方向マスクで pad を落とす ⇔ pad を最初から渡さない）。ホストが列を詰め、出力は
`[1,T,·]` のまま受け取って、必要なら後段でゼロ詰めする（eager 側も pad 行はマスク乗算で
厳密に 0 になる）。同値性は台本が**実測**する（`_static_scheme_evidence` — 全ケースで
「pad して呼んだ結果の先頭 T 行」と「詰めて呼んだ結果」を突き合わせる）。

MUST: full_attention 側のマスクは**渡さない**（`attention_mask` に dict を渡して
`{"full_attention": None, ...}` にする）。マスク無しで呼ぶと ModernBERT は全域 true の帯
マスクを作り、加算型に落とすと `[1,1,Tmax,Tmax]` の**全 0 定数**（f32 1MB）になる — 値としては
恒等なので、初めから渡さない形と厳密に同値で、定数 1MB と 9 層ぶんの mask 加算が消える。
sliding_attention 側（半幅 64 の帯）は本物の帯なので、モデル自身の
`create_bidirectional_sliding_window_mask` で作らせ、Tmax 定数 + `sym_prefix_slice` に
畳ませる（ADR 0010）。

MUST: SDPA は**保存する**（`PRESERVED_OP_PREFIXES_WITH_ATTENTION` — ADR 0023 の融合
attention）。台本ローカルの指定で、既定の分解表には入れない（ADR 0023 決定 5）。

## パッチと参照の順序

`karume.patch_irodori` の差し替えは**プロセス全域**なので、golden の期待値（= パッチ前の
eager 出力）を採り終えるまでパッチを当てない。順序は `export_series` が
`patch_irodori.patches_applied()` で機械的に守る（破れば偽 PASS）。

speaker / duration が要るパッチは 2 つ（complex RoPE の実数化・rank-2 weight RMSNorm の分割
— `patch_irodori` のモジュール docstring）で、`dit` はさらに LowRankAdaLN の weightless RMS
畳み込みを要る。RMSNorm 分割と AdaLN 畳み込みはビット一致が構造的に成り立ち、RoPE の実数化は
**この重みの幾何（head_dim 64）で**ビット一致することが実測なので、`EAGER_EQUIV_ATOL = 0` の
同値検証はテキスト系 3 本と同じ厳しさのまま通る（head_dim が変われば 1 ulp ずれうる —
その時は 0 のまま落ちるので、緩める前に幾何を疑う）。

## 格納 dtype の系列（ADR 0018 / 0019 / 0027 / 0050）

`--dtype f16` / `--dtype i8` はそれぞれ**別系列**（`irodori-v4-small-f16/` /
`irodori-v4-small-i8/`）へ書き出す。同居させると f32 系列の網（実測から導いたターゲット別
tolerance）が圧縮資産へ黙って掛かるため、系列は必ず分ける。丸め（fake-quant）は `load_*` の
直後・**参照 / golden の採取より前**に当てる（`fake_quant` の順序 MUST）。

i8 は per-channel symmetric（ADR 0019）で、丸めと同時に採れる scale 台帳を emit へ渡す
（{@link TARGET_SCALE_SOURCES} が「素のモジュール内 FQN → ラッパ内 FQN」の張り替えを持つ）。

MUST: `--dtype` は emit 専用（`export_sbv2.py` の `--verify` 排他と同じ性格）。この台本は
検証専用モードを持たないので機械的な排他は要らないが、丸めた重みで採った参照を
「パッチ前の参照」以外の用途へ回さない規律は同じ。

## 出力レイアウト

    outputs/series/irodori-v4-small/<target>/model.safetensors      重み・定数 + karume_ir
    outputs/series/irodori-v4-small/<target>/io.<case>.safetensors  入力と torch CPU 期待出力

io のテンソルキー規約は tiny golden / DeBERTa と同じ（`input.<グラフ入力名>` / `output.<位置>`）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.export import Dim

from karume import patch_irodori
from karume.convert import (
    PRESERVED_OP_PREFIXES,
    PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    normalize_boundary_tensor,
)
from karume.ir import IrGraph
from karume.patch_anima import ROPE_BUFFER_NAMES, assert_rope_lifted
from karume.paths import INPUTS_ROOT, SERIES_ROOT
from karume.pipeline import export_to_file
from karume.quantize import QUANT_CHANNEL_AXES, fake_quant_int8, round_weights_to_f16

#: 実重みの置き場（`hf download Aratako/Irodori-TTS-v4-Small` の展開先）。
DEFAULT_MODEL_DIR = INPUTS_ROOT / "irodori" / "v4-small"

#: 書き出せる格納 dtype。`i8` は波 2（ADR 0050 決定 6 の分岐点 = S ドリフトの実測材料）で
#: 足した。**w8a8 は後続の波**（活性量子化は dist の席と判別帯 E2E が要る）。
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "i8")

#: モデル実装（GitHub `Aratako/Irodori-TTS` の clone）の置き場。
DEFAULT_SOURCE_DIR = INPUTS_ROOT / "irodori" / "Irodori-TTS"

#: 実 latent の供給元（`dacvae_host.py` が書く参照音声のホスト前処理 golden）。**そこにある
#: 既製ファイルを読むだけ**で、この台本は DACVAE の重みも encode 経路も引かない
#: （`dacvae_host.py` 側が上流 `encode_waveform` とのビット一致を実測済み）。
REFERENCE_LATENT_DIR = SERIES_ROOT / "dacvae-32dim" / "host"
REFERENCE_LATENT_CASE = "ref-default"
REFERENCE_LATENT_PREFIX = "case."
REFERENCE_LATENT_KEY = "latent"

#: `dacvae_host.py` を回す生成コマンド（実 latent が無いときのエラーにそのまま出す）。
REFERENCE_LATENT_COMMAND = (
    "uv run --with descript-audiotools --with einops --with 'transformers==5.14.1'"
    " python dacvae_host.py"
)

MODEL_FILE = "model.safetensors"
TOKENIZER_FILE = "tokenizer/tokenizer.json"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: チェックポイントの `__metadata__` が持つ 2 本の config（どちらも JSON 文字列）。
#: 実装側の正本は `irodori_tts/inference_runtime.py`。**ここから読むので HF への接続は要らない**。
TEXT_CONFIG_META_KEY = "text_encoder_config_json"
MODEL_CONFIG_META_KEY = "config_json"

#: state_dict の接頭辞（`irodori_tts/model.py` の属性名）。
BACKBONE_PREFIX = "pretrained_text_backbone."
TEXT_PROJ_PREFIX = "text_encoder."
CAPTION_PROJ_PREFIX = "caption_encoder."
SPEAKER_PREFIX = "speaker_encoder."
SPEAKER_NORM_PREFIX = "speaker_norm."
TEXT_NORM_PREFIX = "text_norm."
CAPTION_NORM_PREFIX = "caption_norm."
DURATION_PREFIX = "duration_predictor."

TARGET_BACKBONE = "backbone"
TARGET_TEXT_PROJ = "text-proj"
TARGET_CAPTION_PROJ = "caption-proj"
TARGET_SPEAKER = "speaker"
TARGET_DURATION = "duration"
TARGET_DIT = "dit"
#: 同じ token 列 golden ケースを共有する 3 本（backbone を 2 つの projector が食う鎖）。
TEXT_TARGETS = (TARGET_BACKBONE, TARGET_TEXT_PROJ, TARGET_CAPTION_PROJ)
TARGETS = (*TEXT_TARGETS, TARGET_SPEAKER, TARGET_DURATION, TARGET_DIT)

#: テキスト系 3 本の記号次元名（IR に載る名前）。
TEXT_SYMBOL = "T"

#: `speaker` の記号次元名。参照 latent の**patch 後**の長さで、テキストの T とは別の軸。
SPEAKER_SYMBOL = "S"

#: `dit` の記号次元名（IR に載る名前 — ADR 0047 の `x_t[1,S,32]` の S）。`speaker` の S とは
#: 別のグラフの別の軸で、IR のシンボル名はグラフごとに閉じているので衝突しない。
DIT_SYMBOL = "S"

#: `dit` の **torch 側** `Dim` 名。IR 名（`DIT_SYMBOL`）とは別に持つ。
#:
#: MUST: `"S"` にしてはならない — `Dim.__add__` は派生次元の名前を `sympy.sympify(名前)` で
#: 作るので、`"S"` は sympy の singleton レジストリに解決されて
#: `TypeError: unsupported operand type(s) for +: 'SingletonRegistry' and 'int'` になる。
#: `dit` は `mask` の宣言に派生次元（`S+1519` — ADR 0046）を使う唯一のターゲットなので、
#: ここだけがこの罠を踏む（`speaker` の `Dim("S")` は派生を作らないので無事）。IR 側の名前は
#: `export_to_file(symbol_names=…)` が torch の内部シンボル（`s27` 等）から付け替えるため、
#: torch 名が何であっても IR には `DIT_SYMBOL` が載る。
DIT_TORCH_DIM = "L"

#: 記号次元 T の上限。**text（256）と caption（512）で 1 本に統一**する — backbone は両者で
#: 共有なので、系列を分けると同じ 1.2GB の重みが 2 部できる。畳み込みで焼かれる定数は
#: 半幅 64 の帯マスク（`[1,1,Tmax,Tmax]` f32 = 1MB）と RoPE 表（θ 2 系統 × cos/sin ×
#: `[1,Tmax,64]` = 512KB）だけで、Tmax を 256 → 512 にしても増えるのは 1MB 級（ADR 0010）。
SYM_MAX = 512

#: DACVAE のフレームレート（Hz）。48kHz / hop 1920 = 25。コーデックは別リポ・別重みで、
#: この値はチェックポイントの config に入っていない（`ref_max_seconds` から参照 latent の
#: 上限長を導くのに要る唯一の外部定数）。
CODEC_FRAME_RATE = 25

#: 記号次元の下限。torch.export の 0/1 特殊化を避けるため 2 で運用する（ADR 0010 の 2 点評価と
#: 同じ理由）。
MIN_SYM_LENGTH = 2

#: `dit` の記号次元 S の上限を決める発話長（秒）。実装側の正本は
#: `inference_runtime.SamplingRequest.max_seconds` の既定 30.0（`latent_steps` はこの秒数から
#: `floor(max_seconds × sample_rate / hop_length)` で切られる）。チェックポイントの config には
#: 入っていない（`ref_max_seconds` は**参照 latent** の上限で別物）ので、`CODEC_FRAME_RATE` と
#: 同じく外部定数としてここに置く。上限を 750 に置く判断そのものは ADR 0047。
DIT_MAX_SECONDS = 30.0


def speaker_sym_max(model_config: Mapping[str, Any]) -> int:
    """`speaker` の記号次元 S の上限（参照 latent の patch 後の最大長）。

    `ref_max_seconds`（120s）× 25Hz ÷ `speaker_patch_size`（4）= 750。**チェックポイントの
    config から導出する** — 直書きすると参照長の上限が変わったときに黙って古いままになる。
    焼かれる定数は RoPE 表（`[750,2,64]` f32 = 384KB）だけなので、上限に余裕を持たせても
    コンテナはほとんど増えない。
    """
    seconds = float(model_config["ref_max_seconds"])
    patch = int(model_config["speaker_patch_size"])
    steps = int(seconds * CODEC_FRAME_RATE) // patch
    if steps < MIN_SYM_LENGTH:
        raise SystemExit(f"参照 latent の上限長 {steps} が記号次元の最小 {MIN_SYM_LENGTH} 未満")
    return steps


def dit_sym_max(config: Any) -> int:
    """`dit` の記号次元 S の上限（生成できる latent の最大長 — 30s × 25Hz ÷ patch）。

    `latent_patch_size` は**チェックポイントの config から**取る（x_t の最終次元
    `latent_dim × latent_patch_size` と同じ出どころ）。焼かれる定数は RoPE 表
    （`[750,2,64]` f32 = 384KB）だけなので、上限に余裕を持たせてもコンテナはほとんど増えない。
    """
    patch = int(config.latent_patch_size)
    steps = int(DIT_MAX_SECONDS * CODEC_FRAME_RATE) // patch
    if steps < MIN_SYM_LENGTH:
        raise SystemExit(f"latent の上限長 {steps} が記号次元の最小 {MIN_SYM_LENGTH} 未満")
    return steps


#: 「pad して呼んだ先頭 T 行」と「詰めて呼んだ結果」の一致に許す差（`_static_scheme_evidence`）。
#: 静的方式が**構造的に**成立していれば残るのは softmax の縮約長差による丸めだけで、実測は
#: 1e-5 台。方式が崩れていれば（pad 位置が見えている等）差は状態の値域 O(10) で出る。
STATIC_SCHEME_ATOL = 1e-3

#: ラッパ（+ パッチ）と実モジュールの eager 同値に許す差。**0 = ビット一致を要求する**。
#: 3 つの差（qkv の割り方・全 1 マスクの乗算落とし・全域 true マスクの加算落とし）はいずれも
#: 厳密恒等なので、丸め差が出る余地が無い（実測もビット一致）。0 でない値が出たら、同値の
#: 主張のどれかが崩れている（例: SDPA が mask の有無で別カーネルを選ぶ）— 近似で通さずに
#: 落として、どれが崩れたのかを先に確かめる。
EAGER_EQUIV_ATOL = 0.0

#: text と caption の projector が別物であることを見る下限（`_sanity`）。同じ重みを 2 回
#: 読んでいれば差は厳密に 0 になる。実測は O(1)。
PROJECTOR_DIVERGENCE_MIN = 1e-2

#: `text_norm` と `caption_norm` が別物であることを見る下限（`_norm_divergence`）。同じ重みを
#: 2 回読んでいれば差は厳密に 0 になる。`caption-proj` の第 2 出力の契約を守る唯一の門。
NORM_DIVERGENCE_MIN = 1e-3

#: 長文ケースの本文（1 段落 ≈ 200 トークン）。sliding_window の半幅 64 を十分に超える T で
#: 帯マスクが**本当に帯として**効いていることを見るためのもの（T ≤ 129 では帯が全域に
#: 近くなり、窓の取り違えが golden に現れない）。
LONG_PASSAGE_A = (
    "秋の朝、駅前の商店街はまだ静かで、シャッターの下りた店の前を、通勤の人がまばらに"
    "歩いていた。パン屋の換気口から流れてくる甘い匂いだけが、これから一日が始まることを"
    "教えている。私は改札の手前で立ち止まり、鞄の中をもう一度だけ確かめた。定期券、財布、"
    "それから昨日の夜にようやく書き上げた原稿の束。指先にざらついた紙の感触が伝わってきて、"
    "そこで初めて、これが現実の予定なのだという実感が戻ってきた。ホームに上がると、向かいの"
    "線路の向こうに、朝日を受けた高いビルが並んでいるのが見えた。ガラスの壁面がひとつずつ"
    "順番に光り始め、まるで街全体がゆっくりと目を覚ましていくようだった。遠くから電車の"
    "近づいてくる音が聞こえてくる。私は深く息を吸い込み、今日これから会う人たちのことを"
    "順番に思い浮かべた。"
)

#: 長文ケースの 2 段落目と 3 段落目。`LONG_PASSAGE_A` と繋いで T を 400 トークン級まで
#: 伸ばし、宣言上限 Tmax = 512 の近傍を踏む（`sym_prefix_slice` が上限に寄った長さでも
#: 正しく切れることの確認）。
LONG_PASSAGE_B = (
    "打ち合わせの場所は、線路沿いの古い建物の三階だった。エレベーターが無いので階段を"
    "上がるしかなく、途中の踊り場で一度だけ立ち止まって呼吸を整えた。扉を開けると、"
    "細長い部屋の奥に大きな机がひとつ置かれていて、その上には前回の打ち合わせで使った"
    "資料がそのまま積まれていた。窓は北向きで、季節に関わらず光は柔らかい。"
    "担当の人はまだ来ていなかったので、私は椅子に腰を下ろし、持ってきた原稿をもう一度"
    "読み返した。読み返すたびに直したい箇所が見つかるのは毎回のことで、けれども今日は、"
    "書いたときに考えていたことがそのまま残っているように思えた。階段を上がってくる足音が"
    "近づいてきて、私は原稿を閉じ、顔を上げて扉のほうを見た。"
)

LONG_PASSAGE_C = (
    "話が一区切りついたのは、昼を少し回ったころだった。窓の外では雲が薄く広がっていて、"
    "先ほどまで机の上に落ちていた光の四角形が、いつの間にか輪郭を失っていた。担当の人は"
    "資料の余白に細かく書き込みを入れながら、次の段取りを順番に確認していった。締切、"
    "分量、それから読み手として想定している人の年齢層。ひとつずつ言葉にしていくと、"
    "頭の中で漠然としていた輪郭が、少しずつ具体的な形になっていくのが分かった。帰り際、"
    "階段の途中で振り返ると、廊下の奥の窓から差し込む光が、床の埃をゆっくりと照らしていた。"
    "外に出ると風が思っていたより冷たく、私は上着の襟を立てて、来たときと同じ道を駅へ向かって"
    "歩き始めた。商店街はすでに人で埋まっていて、朝の静けさはどこにも残っていなかった。"
)

#: golden の固定ケース `(名前, 種別, 本文)`。**種別**は静的方式の実測（`_static_scheme_evidence`）
#: で使う pad 長（text = `max_text_len` / caption = `max_caption_len`）を決めるためだけの区別で、
#: グラフには影響しない（backbone は両者で共有）。本文は公式モデルカードの実例（読み上げ文 /
#: Voice Design caption / 絵文字入り）と、上の長文 3 段落から採る。T は 7 / 13 / 22 / 29 /
#: 144 / 404 で、宣言上限 Tmax = 512 に対して短長の両端を踏む。
GOLDEN_CASES: tuple[tuple[str, str, str], ...] = (
    ("text-short", "text", "今日は近くの店まで歩いて行きました。"),
    (
        "text-formal",
        "text",
        "本日はお越しいただき、誠にありがとうございます。どうぞごゆっくりお過ごしください。",
    ),
    (
        "text-emoji",
        "text",
        "あははっ🤭、それ本当に言ってるの？…😮‍💨まぁ、君らしいけどね。",
    ),
    (
        "caption-ja",
        "caption",
        "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
    ),
    ("text-long", "text", LONG_PASSAGE_A),
    ("caption-long", "caption", LONG_PASSAGE_A + LONG_PASSAGE_B + LONG_PASSAGE_C),
)

#: `speaker` の**合成** golden ケース `(名前, patch 後トークン数 S, 乱数 seed)`。
#:
#: 合成（決定的 seed の `torch.randn`）を残すのは、この 5 本が**長さの被覆**（S = 2 …
#: 宣言上限 750）の担い手だから — 実 latent 1 本では 47 行しか取れず、上限も下限も踏めない。
#: 値の性格（実音声の DACVAE latent の値域）は {@link SPEAKER_REAL_CASES} が受け持つ。
#:
#: 長さは秒数から採る: 25Hz ÷ `speaker_patch_size` 4 = 6.25 token/s なので 1s / 5s / 30s に
#: 相当する 6 / 31 / 187 と、**宣言上限そのもの**（120s = 750）、および記号次元の下限 2。
#: 上限と下限の両端を踏むのは、RoPE 表の `sym_prefix_slice` が端で崩れないことを見るため。
SPEAKER_CASES: tuple[tuple[str, int, int], ...] = (
    ("ref-min", 2, 101),
    ("ref-1s", 6, 102),
    ("ref-5s", 31, 103),
    ("ref-30s", 187, 104),
    ("ref-max", 750, 105),
)

#: `speaker` の**実 latent** golden ケース `(名前, patch 後トークン数 S〈None = 全長〉)`。
#:
#: 供給元は公式サンプルの参照音声（7.6 秒 = 190 フレーム）を上流の決定的 encode に通した
#: {@link REFERENCE_LATENT_KEY} で、patch（`speaker_patch_size` 4・端は上流が捨てる）後は
#: 47 行。合成の標準正規とは値域も相関構造も違うので、**tolerance の根拠を実運用の値域と
#: 対応させる**ためにこの 2 本を足す（`export_dacvae.py` の `DECODER_CASES` と同じ理由）。
#:
#: 短尺側を 6 に採るのは、**合成の `ref-1s` と同じ S** で並べるため — 同じ長さで合成と実の
#: 誤差を直接比べられる形にしておくと、tolerance が「長さ」で決まっているのか「値域」で
#: 決まっているのかが表から読める。
SPEAKER_REAL_CASES: tuple[tuple[str, int | None], ...] = (
    ("ref-real-short", 6),
    ("ref-real-full", None),
)

#: `duration` の golden ケース `(名前, text ケース名, has_speaker, has_caption)`。
#:
#: text 系列は**上流 `text-proj` の torch 期待値そのもの**を食わせる（鎖 — projector の
#: golden と同じ値がそのまま duration の入力になる）。speaker / caption のベクトルも
#: `speaker` ターゲットの torch 期待値と `caption-proj` の torch 期待値から、実装の
#: `_prepend_masked_mean_token` / `_caption_vec` を**呼んで**作る（式を写さない）。
#:
#: 4 通りの `(has_speaker, has_caption)` を全て踏む — グラフ内に残した 2 本の `where`
#: （null ベクトル選択）は、片側だけだと恒真化して取り違えが出ない。
DURATION_CASES: tuple[tuple[str, str, bool, bool], ...] = (
    ("dur-both", "text-short", True, True),
    ("dur-speaker-only", "text-formal", True, False),
    ("dur-caption-only", "text-emoji", False, True),
    ("dur-neither", "caption-ja", False, False),
    ("dur-long", "text-long", True, True),
)

#: `duration` の speaker / caption ベクトルの供給元ケース。**長さの違う 2 本を選ぶ**理由は
#: 無い（ベクトルは長さに依らない `[1,768]` / `[1,512]`）ので、それぞれ 1 本に固定する。
DURATION_SPEAKER_SOURCE = "ref-5s"
DURATION_CAPTION_SOURCE = "caption-ja"

#: `dit` の条件 state の供給元ケース（鎖 — 上流ターゲットの torch 期待値そのもの）。
#: text は Tmax = 256 に対して長め（T=144）、caption は本物の Voice Design 文（T=22）、
#: speaker は 5s 相当（S=31 → 平均トークン前置で 32）。
DIT_TEXT_SOURCE = "text-long"
DIT_CAPTION_SOURCE = "caption-ja"
DIT_SPEAKER_SOURCE = "ref-5s"

#: `dit` の CFG 変種名（`rf.py` の `independent_names` と同じ綴り）。`None` = cond。
DIT_UNCOND_VARIANTS = ("text", "speaker", "caption")

#: `dit` の golden ケース `(名前, latent 長 S, 乱数 seed, t, uncond 区間)`。
#:
#: `x_t` は決定的 seed の標準正規（推論の x_t は t=1 で純ノイズ・以降もノイズ寄りなので、
#: 合成で値域の性格が大きく外れない — 実 z の値域がほぼ単位分散であることは speaker 実 latent
#: ケースの導入時に実測済みで、この選択を裏付けている）。
#:
#: ケース軸は 4 本:
#:
#: - **cond / uncond 3 変種** — uncond は **cond の state のままマスクの当該区間だけ 0**。
#:   参照値は上流の uncond（state 0 + マスク 0）で採るので、`EAGER_EQUIV_ATOL = 0` の
#:   同値がそのまま ADR 0047 決定 1 の実証になる。取り違え防止は
#:   `_dit_uncond_divergence`（cond と実際に違う値が出ることを毎回実測）。
#: - **t の 2 点**（0.9 / 0.3）— 上流の CFG は t ∈ [0.5, 1.0] だけに掛かるので、その内外を
#:   1 点ずつ踏む。`t_embed` はホスト生成なので、取り違えは値としてここに出る。
#: - **S の短長** — 記号次元の下限 2・1s 相当の 25・**宣言上限そのものの 750**。
#:   750 は中間 `scores[1,20,750,2269]` が 130MB になる点で、実 GPU の門としても重い側。
#: - uncond 3 変種は cond の 1s ケースと **x_t / t / 条件 state を共有**する（差はマスクだけ）。
DIT_CASES: tuple[tuple[str, int, int, float, str | None], ...] = (
    ("dit-cond-min", 2, 301, 0.9, None),
    ("dit-cond-1s", 25, 302, 0.9, None),
    ("dit-cond-late", 25, 303, 0.3, None),
    ("dit-uncond-text", 25, 302, 0.9, "text"),
    ("dit-uncond-speaker", 25, 302, 0.9, "speaker"),
    ("dit-uncond-caption", 25, 302, 0.9, "caption"),
    ("dit-cond-max", 750, 304, 0.9, None),
)

#: cond と uncond 3 変種の出力が**互いに**違うことを見る下限（`_dit_uncond_divergence`）。
#:
#: MUST: 恒真にしない — マスクの区間割り（self S / text 256 / speaker 751 / caption 512 の
#: 並びとオフセット）が崩れても、ラッパと参照が**同じ**崩れ方をすれば golden は一致する
#: （例: どの変種も既に pad だった位置しか 0 にしていない = 全変種が cond と同値／3 変種が
#: 同じ区間を落としている）。4 本の出力の**総当たり最小差**を毎回実測して、その形を落とす。
DIT_UNCOND_DIVERGENCE_MIN = 1e-3


class IrodoriSource:
    """`irodori_tts` パッケージから取り出す実装（`sys.path` 追加で import する）。

    MUST: import は**実装 clone から**行う（写しを台本に持たない）。projector は 4 行の
    素直な式だが、写せば「上流が変わっても台本だけ古いまま黙って通る」形になる。
    """

    def __init__(self, source_dir: Path) -> None:
        model_py = source_dir / "irodori_tts" / "model.py"
        if not model_py.is_file():
            raise SystemExit(
                f"モデル実装が見つからない: {model_py}"
                "（`git clone https://github.com/Aratako/Irodori-TTS` の展開先を"
                " --source-dir に指定する）"
            )
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from irodori_tts.config import ModelConfig
        from irodori_tts.model import (
            DurationPredictor,
            PretrainedConditionProjector,
            PretrainedTextBackbone,
            ReferenceLatentEncoder,
            RMSNorm,
            TextToLatentRFDiT,
            get_timestep_embedding,
            patch_sequence_with_mask,
        )
        from irodori_tts.text_normalization import normalize_text

        self.backbone_cls = PretrainedTextBackbone
        self.projector_cls = PretrainedConditionProjector
        self.speaker_cls = ReferenceLatentEncoder
        self.duration_cls = DurationPredictor
        self.rms_norm_cls = RMSNorm
        self.model_config_cls = ModelConfig
        self.dit_cls = TextToLatentRFDiT
        #: 平均トークンの前置は `TextToLatentRFDiT` の staticmethod。台本は式を写さずに呼ぶ。
        self.prepend_masked_mean_token = TextToLatentRFDiT._prepend_masked_mean_token
        #: 参照 latent の束ね（`[T,32]` → `[T//4,128]`・端は捨てる）。**式を写さず呼ぶ** —
        #: 端の扱い（切り捨て / 切り上げ）が上流で変われば、写した式は黙って古いまま通る。
        self.patch_sequence_with_mask = patch_sequence_with_mask
        #: `t_embed` のホスト生成（sin/cos 3 行）。**式を写さず呼ぶ** — θ の割り方が上流で
        #: 変われば、写した式は黙って古いまま通る。
        self.timestep_embedding = get_timestep_embedding
        self.normalize_text = normalize_text

    def model_config(self, raw: Mapping[str, Any]) -> Any:
        """チェックポイントの `config_json` から `ModelConfig` を組む。

        `config_json` には学習側の値（`max_text_len` / `max_caption_len` / `ref_max_seconds`）も
        混ざっているので、dataclass のフィールドだけを拾う。落ちた 3 キーは捨てるのではなく、
        呼び出し側が生 dict から直接読む（`speaker_sym_max` / `_static_scheme_evidence`）。
        """
        names = {field.name for field in dataclasses.fields(self.model_config_cls)}
        return self.model_config_cls(**{key: value for key, value in raw.items() if key in names})


def read_configs(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """チェックポイントの `__metadata__` から `(text_encoder_config, model_config)` を読む。

    MUST: HF から config を引き直さない — チェックポイントに埋まっている dict が、この重みが
    実際に構成されたときの形の正本（`text_encoder_revision` 込み）。
    """
    with safe_open(str(model_dir / MODEL_FILE), framework="pt") as handle:
        metadata = handle.metadata()
    if metadata is None:
        raise SystemExit(f"{model_dir / MODEL_FILE} に __metadata__ が無い")
    missing = [key for key in (TEXT_CONFIG_META_KEY, MODEL_CONFIG_META_KEY) if key not in metadata]
    if missing:
        raise SystemExit(f"{model_dir / MODEL_FILE} の __metadata__ に {missing} が無い")
    return (
        json.loads(metadata[TEXT_CONFIG_META_KEY]),
        json.loads(metadata[MODEL_CONFIG_META_KEY]),
    )


def _sub_state(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    picked = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not picked:
        raise SystemExit(f"チェックポイントに接頭辞 '{prefix}' のテンソルが 1 本も無い")
    return picked


def load_backbone(
    source: IrodoriSource,
    state: Mapping[str, torch.Tensor],
    text_config: Mapping[str, Any],
) -> nn.Module:
    """`PretrainedTextBackbone` を**重み読み込み無し**で構成し、チェックポイントの重みを載せる。

    `load_pretrained_weights=False` にすると `no_init_weights` + `from_config` の経路になり、
    HF への接続も乱数初期化も走らない（載せる重みは全て state_dict 側から来る）。
    """
    backbone = source.backbone_cls(
        str(text_config.get("_name_or_path", "")),
        config_dict=dict(text_config),
        load_pretrained_weights=False,
    )
    backbone.load_state_dict(_sub_state(state, BACKBONE_PREFIX), strict=True)
    return backbone.eval()


def load_projector(
    source: IrodoriSource,
    state: Mapping[str, torch.Tensor],
    model_config: Mapping[str, Any],
    prefix: str,
    backbone_dim: int,
    output_dim: int,
) -> nn.Module:
    """`PretrainedConditionProjector` を config どおりに組み、指定接頭辞の重みを載せる。

    MUST: 入力次元は backbone から受け取る（写しを持たない）。`load_state_dict(strict=True)` が
    形の食い違いを落とすので、取り違えは黙って通らない。
    """
    projector = source.projector_cls(
        backbone_dim,
        output_dim,
        projector_type=str(model_config["pretrained_projector_type"]),
        hidden_ratio=float(model_config["pretrained_projector_hidden_ratio"]),
        dropout=0.0,
        norm_eps=float(model_config["norm_eps"]),
    )
    projector.load_state_dict(_sub_state(state, prefix), strict=True)
    return projector.eval()


def load_rms_norm(
    source: IrodoriSource,
    state: Mapping[str, torch.Tensor],
    prefix: str,
    dim: int,
    norm_eps: float,
) -> nn.Module:
    """接頭辞 1 本ぶんの `RMSNorm` を組んで重みを載せる（`load_state_dict(strict=True)` が門）。"""
    norm = source.rms_norm_cls(dim, eps=norm_eps)
    norm.load_state_dict(_sub_state(state, prefix), strict=True)
    return norm.eval()


def load_speaker_encoder(
    source: IrodoriSource, state: Mapping[str, torch.Tensor], config: Any
) -> nn.Module:
    """`ReferenceLatentEncoder` を config どおりに組み、チェックポイントの重みを載せる。"""
    encoder = source.speaker_cls(config)
    encoder.load_state_dict(_sub_state(state, SPEAKER_PREFIX), strict=True)
    return encoder.eval()


def load_duration_predictor(
    source: IrodoriSource, state: Mapping[str, torch.Tensor], config: Any
) -> nn.Module:
    """`DurationPredictor` を config どおりに組み、チェックポイントの重みを載せる。

    MUST: 引数は `TextToLatentRFDiT.__init__` と同じ並びで config から渡す（`model.py` の
    構成を写している唯一の箇所 — dataclass の名前が変われば `AttributeError` で落ちる）。
    """
    predictor = source.duration_cls(
        text_dim=config.text_dim,
        aux_dim=config.duration_aux_dim,
        hidden_dim=config.duration_hidden_dim,
        layers=config.duration_layers,
        dropout=config.duration_dropout,
        speaker_dim=config.speaker_dim if config.use_speaker_condition_resolved else None,
        speaker_fusion=config.duration_speaker_fusion,
        caption_dim=config.caption_dim_resolved if config.use_caption_condition else None,
        caption_fusion=config.duration_caption_fusion,
        caption_pooling=config.duration_caption_pooling,
        attention_heads=config.duration_attention_heads,
        norm_eps=config.norm_eps,
        architecture=config.duration_architecture,
        token_init_frames=config.duration_token_init_frames,
    )
    predictor.load_state_dict(_sub_state(state, DURATION_PREFIX), strict=True)
    return predictor.eval()


def load_dit(
    source: IrodoriSource,
    state: Mapping[str, torch.Tensor],
    config: Any,
    text_config: Mapping[str, Any],
) -> nn.Module:
    """`TextToLatentRFDiT` **丸ごと**を組み、チェックポイントの全テンソルを載せる。

    DiT だけを部分的に組み直さないのは、golden の参照値を実装の
    `forward_with_encoded_conditions`（= 上流の正本経路）から採るため。`strict=True` の
    全件ロードが「714 本を 1 本残らず消費した」ことの門にもなる。

    backbone は `load_pretrained_backbone_weights=False` で構成する（HF への接続も乱数初期化も
    走らない — 載せる重みは全て state_dict 側から来る）。**このターゲットは backbone を
    グラフに載せない**（`DitGraph` が持つのは DiT 本体の部分木だけ）ので、ここで構成された
    backbone は参照の一部にもならない。
    """
    model = source.dit_cls(
        config,
        pretrained_backbone_config=dict(text_config),
        load_pretrained_backbone_weights=False,
    )
    model.load_state_dict(state, strict=True)
    return model.eval()


class BackboneGraph(nn.Module):
    """`PretrainedTextBackbone.forward` の export 用ラッパ（マスクを実行時入力に持たない）。

    元の forward との差は 2 点で、どちらも**全 1 マスクのとき厳密恒等**（モジュール docstring
    の MUST — `_check_wrapper_equivalence` が実重みで毎回実測する）:

    - 出力の `state * mask` を落とす（`×1`）
    - `attention_mask` に dict を渡し、full_attention 側を `None` にする（全域 true の帯マスク
      = 加算型では全 0 なので、加算そのものが恒等）
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        # inv_freq がバッファのままだと定数畳み込みの葉にならず、sin / cos が IR に残る。
        assert_rope_lifted(backbone.backbone, "irodori text backbone")
        self.model = backbone.backbone

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        from transformers.masking_utils import create_bidirectional_sliding_window_mask

        model = self.model
        # 帯マスクの生成に要るのは shape / dtype / device だけ（transformers 側の docstring）。
        # 実体を作らせないために、hidden 幅 1 のゼロを渡す（定数畳み込みの葉になる）。
        metadata = torch.zeros(input_ids.shape[0], input_ids.shape[1], 1, dtype=torch.float32)
        masks = {
            "full_attention": None,
            "sliding_attention": create_bidirectional_sliding_window_mask(
                config=model.config, inputs_embeds=metadata, attention_mask=None
            ),
        }
        return model(input_ids=input_ids, attention_mask=masks).last_hidden_state


class ProjectorGraph(nn.Module):
    """`PretrainedConditionProjector.forward` の export 用ラッパ（backbone を含まない）。

    元の forward との差は 2 点で、どちらも**全 1 マスクのとき厳密恒等**:

    - backbone 呼び出しを外へ出す（`state` を引数で受ける）
    - 出力の `projected * mask` を落とす（`×1`）

    `residual_mlp` 以外の projector 型は Irodori v4-Small に現れないので受けない
    （`export_series` が config を見て fail loudly にする）。
    """

    def __init__(self, projector: nn.Module) -> None:
        super().__init__()
        self.projector = projector

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projector = self.projector
        projected = projector.projector(hidden)
        residual = projector.residual_norm(hidden)
        residual = torch.nn.functional.silu(projector.residual_up(residual))
        return projected + projector.residual_down(residual)


class CaptionProjectorGraph(nn.Module):
    """caption 側 projector の export 用ラッパ（**2 出力** — モジュール docstring の G1b 節）。

    第 1 出力は {@link ProjectorGraph} そのもの（生の projector 出力）で、第 2 出力は
    そこへ `caption_norm`（RMSNorm 512）を掛けた系列。`duration` が要る `caption_vec` の
    masked mean をホストが採れるようにするためだけに足してある。

    MUST: 第 1 出力の計算は `ProjectorGraph` を**そのまま使う**（式を写さない）— 写すと
    text 側との差が黙って入りうるし、第 1 出力のビット一致（旧 golden との突合）が
    「同じ式を 2 箇所に書いた」ことの確認に堕ちる。
    """

    def __init__(self, projector: nn.Module, caption_norm: nn.Module) -> None:
        super().__init__()
        self.projection = ProjectorGraph(projector)
        self.caption_norm = caption_norm

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(hidden)
        return projected, self.caption_norm(projected)


class SpeakerGraph(nn.Module):
    """`ReferenceLatentEncoder.forward` + `speaker_norm` の export 用ラッパ。

    元の 2 段（`encode_conditions`）との差は 2 点で、どちらも**全 1 マスクのとき厳密恒等**:

    - `x * mask_f` を 3 箇所（in_proj 直後・各ブロック後・出力）落とす（`×1`）
    - `SelfAttention` の `key_mask` を `None` にする（全 1 マスクの SDPA は加算バイアスが
      全 0 なので `scores + 0` の恒等）

    RoPE 表は **Tmax（= S の宣言上限）で焼いた実数形定数**を渡す（`patch_irodori` の
    `real_pair_rope_table`）。`SelfAttention` 側の `freqs_cis[:seq_len]` が記号 prefix
    スライスになり、定数畳み込みは ADR 0010 の経路にそのまま乗る（backbone の帯マスク /
    RoPE 表と同じ扱い）。**グラフ入力へ昇格しない**のは、表が 384KB しかなく、
    「モデルファイル 1 個で完結」を崩す理由が無いから（ADR 0010 の代替案 2）。
    """

    def __init__(self, encoder: nn.Module, speaker_norm: nn.Module, sym_max: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.speaker_norm = speaker_norm
        # 素の属性（lifted tensor constant）にする。バッファ/パラメータは定数畳み込みの葉に
        # ならず、cos / sin がそのまま IR に残る（patch_anima.lift_rope_buffers と同じ理由）。
        self.rope_table = patch_irodori.real_pair_rope_table(encoder.head_dim, sym_max)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        x = encoder.in_proj(latent)
        x = x / 6.0
        for block in encoder.blocks:
            x = block(x, mask=None, freqs_cis=self.rope_table)
        return self.speaker_norm(x)


class DurationGraph(nn.Module):
    """`text_norm` + `DurationPredictor`（token-sum 形）の export 用ラッパ。

    元の `predict_duration_log_frames` との差は 3 点で、いずれも入力契約（B=1・全 1 マスク・
    テキストは 1 token 以上）の下で厳密恒等:

    - `_safe_attention_mask` を落とす（全 1 マスクでは入力をそのまま返す恒等）
    - `token_frames * text_mask` を落とす（`×1`）
    - `speaker_state[:,0]` / caption の masked mean を**ホストが切り出した結果**で受ける
      （`null_*` の選択は `has_*` の bool 入力とともにグラフに残す — モジュール docstring）

    `aux_features` は token-sum 形が一切読まないので受けない（`_duration_aux_is_inert` が
    「読まれていない」ことを毎回実測する）。
    """

    def __init__(self, predictor: nn.Module, text_norm: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor
        self.text_norm = text_norm

    def forward(
        self,
        text_state: torch.Tensor,
        speaker_vec: torch.Tensor,
        has_speaker: torch.Tensor,
        caption_vec: torch.Tensor,
        has_caption: torch.Tensor,
    ) -> torch.Tensor:
        predictor = self.predictor
        speaker = torch.where(has_speaker, speaker_vec, predictor.null_speaker)
        caption = torch.where(has_caption, caption_vec, predictor.null_caption)
        hidden = predictor.token_input_proj(self.text_norm(text_state))
        for block in predictor.token_blocks:
            hidden = block(hidden, cond=speaker, caption_cond=caption)
        logits = predictor.token_out_proj(predictor.token_out_norm(hidden)).squeeze(-1)
        frames = torch.nn.functional.softplus(logits.float())
        return torch.log1p(frames.sum(dim=1).clamp_min(0.0))


class DitGraph(nn.Module):
    """DiT 1 step（`forward_with_encoded_conditions`）の export 用ラッパ。

    元の `forward_with_encoded_conditions` との差は 4 点で、いずれも入力契約（B=1・条件は
    Tmax 右 pad・マスクは self / text / speaker / caption の順に連結）の下で厳密恒等:

    - `t_embed` を**引数で受ける**（`get_timestep_embedding` はホスト — `cos` が語彙に無い）
    - `text_norm` / `caption_norm` を**このグラフが掛ける**（入力は projector の生の出力。
      `rms_norm` は行ごとの縮約なので、pad の前に掛けても後に掛けても先頭行は同値。
      pad 行は 0 で、`rms_norm(0)` も厳密に 0）
    - `JointAttention` の**マスク 4 本の連結を引数 1 本で受ける**（上流の
      `torch.cat(context_masks, dim=1)[:, None, None, :]` と同じもの）。K/V の連結は
      **記号軸 cat**（`S+1519` — ADR 0046）で、条件側の 3 本は静的軸 cat に畳んである
      （`cat` は結合的で、どちらの括り方でも同じ要素が同じ順に並ぶ）
    - `nn.Dropout(p=0.0)` を落とす（eval では厳密恒等）

    K/V の射影（`project_context_kv`）は**毎 forward グラフ内で計算する**（G4 の畳み込み —
    ADR 0047 決定 3）。射影も RoPE も q/k ノルムも実モジュールのメソッドをそのまま呼ぶので、
    このラッパが写しているのは「連結の順序」と「residual の組み立て」だけ。

    RoPE 表は `speaker` と同じく **Smax で焼いた実数形定数**（`patch_irodori` の
    `real_pair_rope_table`）で、`[:seq_len]` の記号 prefix スライスが ADR 0010 の経路に乗る。
    """

    def __init__(self, model: nn.Module, sym_max: int) -> None:
        super().__init__()
        self.cond_module = model.cond_module
        self.in_proj = model.in_proj
        self.blocks = model.blocks
        self.out_norm = model.out_norm
        self.out_proj = model.out_proj
        self.text_norm = model.text_norm
        self.caption_norm = model.caption_norm
        # 素の属性（lifted tensor constant）にする — SpeakerGraph と同じ理由。
        self.rope_table = patch_irodori.real_pair_rope_table(model.head_dim, sym_max)

    def forward(
        self,
        x_t: torch.Tensor,
        t_embed: torch.Tensor,
        mask: torch.Tensor,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor,
        caption_state: torch.Tensor,
    ) -> torch.Tensor:
        cond_embed = self.cond_module(t_embed)[:, None, :]
        text = self.text_norm(text_state)
        caption = self.caption_norm(caption_state)
        x = self.in_proj(x_t)
        # 上流の `_rope_freqs(x.shape[1])` と同じ長さまで切る（`JointAttention` 側の
        # `freqs_cis[:seq_len]` は、この時点で恒等になる）。
        freqs = self.rope_table[: x.shape[1]]
        for block in self.blocks:
            h, attention_gate = block.attention_adaln(x, cond_embed)
            x = x + attention_gate * self._attention(
                block.attention, h, text, speaker_state, caption, mask, freqs
            )
            h, mlp_gate = block.mlp_adaln(x, cond_embed)
            x = x + mlp_gate * block.mlp(h)
        return self.out_proj(self.out_norm(x))

    @staticmethod
    def _attention(
        attention: nn.Module,
        x: torch.Tensor,
        text: torch.Tensor,
        speaker: torch.Tensor,
        caption: torch.Tensor,
        mask: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """`JointAttention.forward` の同値実装（連結済み K/V + 連結済みマスク）。"""
        bsz, seq_len, _ = x.shape
        heads, head_dim = attention.heads, attention.head_dim
        q = attention.wq(x).reshape(bsz, seq_len, heads, head_dim)
        k_self = attention.wk(x).reshape(bsz, seq_len, heads, head_dim)
        v_self = attention.wv(x).reshape(bsz, seq_len, heads, head_dim)
        projected = attention.project_context_kv(
            text_context=text, speaker_context=speaker, caption_context=caption
        )
        if len(projected) != 6:
            raise AssertionError(
                f"context KV が {len(projected)} 本（text / speaker / caption の 3 条件 = 6 本）"
                " — この重みは 3 条件とも有効なはずで、条件の有無が config と食い違っている"
            )
        k_text, v_text, k_speaker, v_speaker, k_caption, v_caption = projected

        q = attention.q_norm(q)
        k_self = attention.k_norm(k_self)
        q = attention._apply_rotary_half(q, freqs)
        k_self = attention._apply_rotary_half(k_self, freqs)

        context_k = torch.cat([k_text, k_speaker, k_caption], dim=1)
        context_v = torch.cat([v_text, v_speaker, v_caption], dim=1)
        k = torch.cat([k_self, context_k], dim=1)
        v = torch.cat([v_self, context_v], dim=1)

        y = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, is_causal=False
        ).transpose(1, 2)
        y = y.reshape(bsz, seq_len, attention.dim)
        y = y * torch.sigmoid(attention.gate(x))
        return attention.wo(y)


def golden_case_body(name: str, kind: str, body: str, normalize_text: Any) -> str:
    """golden ケースの本文へ**種別ごとの**前処理を当てる（上流 `_synthesize` と同じ）。

    MUST: `normalize_text` は **text 専用**。上流 `inference_runtime._synthesize` が caption に
    掛けるのは `str(...).strip()` だけなので、caption へ正規化を足すと外側括弧の剥がし・NFKC・
    記号削除のぶんだけ conditioning が黙って別物になる（同じ理由で `irodori_pipeline.py` の
    `_packed_caption_ids` も strip だけを掛ける）。
    """
    if kind == "text":
        prepared = normalize_text(body).strip()
    elif kind == "caption":
        prepared = body.strip()
    else:
        raise SystemExit(f"{name}: 種別 {kind!r} は text / caption のどちらでもない")
    if not prepared:
        raise SystemExit(f"{name}: 前処理後の本文が空")
    return prepared


def build_cases(
    model_dir: Path, source: IrodoriSource, text_config: Mapping[str, Any], sym_max: int
) -> tuple[tuple[str, str, torch.Tensor], ...]:
    """golden ケースの `(名前, 種別, input_ids)`。

    前処理は実装の正本（`inference_runtime._synthesize`）と同じ順序:
    種別ごとの前処理（{@link golden_case_body}）→ 特殊トークン無しでトークナイズ → BOS 前置。
    右詰め pad は**しない**（静的方式ではホストが列を詰める — モジュール docstring）。
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
    bos_id = int(text_config["bos_token_id"])
    cases: list[tuple[str, str, torch.Tensor]] = []
    for name, kind, body in GOLDEN_CASES:
        prepared = golden_case_body(name, kind, body, source.normalize_text)
        ids = [bos_id, *tokenizer.encode(prepared, add_special_tokens=False).ids]
        if not 2 <= len(ids) <= sym_max:
            raise SystemExit(f"{name}: T={len(ids)} が記号次元の範囲 [2, {sym_max}] の外")
        cases.append((name, kind, torch.tensor([ids], dtype=torch.int64)))
    return tuple(cases)


def build_speaker_cases(latent_dim: int, sym_max: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """`speaker` の**合成** golden ケース `(名前, 参照 latent)`（{@link SPEAKER_CASES} どおり）。

    値は決定的 seed の標準正規（表の注記 — 長さの被覆はこちらが担う）。
    """
    cases: list[tuple[str, torch.Tensor]] = []
    for name, length, seed in SPEAKER_CASES:
        if not MIN_SYM_LENGTH <= length <= sym_max:
            raise SystemExit(
                f"{name}: S={length} が記号次元の範囲 [{MIN_SYM_LENGTH}, {sym_max}] の外"
            )
        generator = torch.Generator().manual_seed(seed)
        cases.append((name, torch.randn(1, length, latent_dim, generator=generator)))
    return tuple(cases)


def build_real_speaker_cases(
    patch_sequence: Any,
    latent_dim: int,
    patch_size: int,
    patched_dim: int,
    sym_max: int,
    *,
    latent_dir: Path = REFERENCE_LATENT_DIR,
) -> tuple[tuple[str, torch.Tensor], ...]:
    """`speaker` の**実 latent** golden ケース（{@link SPEAKER_REAL_CASES} の表どおり）。

    供給元は `dacvae_host.py` が書いた既製の `[1,190,32]`。patch は**上流の
    `patch_sequence_with_mask` を呼ぶ**（`patch_sequence` に渡ってくる） — 端の切り捨ての
    仕方を写すと、上流が変わっても台本だけ古いまま黙って通る。

    MUST: 実 latent が無い環境では**落とす**（合成で代替しない） — 代替すると、この 2 本が
    受け持っている「値域が実運用のもの」という性質が黙って消え、tolerance の根拠が壊れる。
    """
    path = latent_dir / f"{REFERENCE_LATENT_PREFIX}{REFERENCE_LATENT_CASE}{IO_SUFFIX}"
    if not path.is_file():
        raise SystemExit(f"実 latent が無い: {path}（`{REFERENCE_LATENT_COMMAND}` で作る）")
    tensors = load_file(str(path))
    if REFERENCE_LATENT_KEY not in tensors:
        raise SystemExit(f"{path} に '{REFERENCE_LATENT_KEY}' が無い")
    latent = tensors[REFERENCE_LATENT_KEY].to(torch.float32)
    if latent.ndim != 3 or latent.shape[0] != 1 or int(latent.shape[2]) != latent_dim:
        raise SystemExit(f"実 latent の shape {tuple(latent.shape)} が [1,S,{latent_dim}] でない")
    mask = torch.ones(latent.shape[:2], dtype=torch.bool)
    patched, _patched_mask = patch_sequence(latent, mask, patch_size)
    if int(patched.shape[2]) != patched_dim:
        raise SystemExit(
            f"patch 後の次元 {int(patched.shape[2])} が speaker の入力次元 {patched_dim} と違う"
        )
    available = int(patched.shape[1])

    cases: list[tuple[str, torch.Tensor]] = []
    for name, length in SPEAKER_REAL_CASES:
        want = available if length is None else int(length)
        if not MIN_SYM_LENGTH <= want <= sym_max:
            raise SystemExit(
                f"{name}: S={want} が記号次元の範囲 [{MIN_SYM_LENGTH}, {sym_max}] の外"
            )
        if want > available:
            raise SystemExit(f"{name}: S={want} が実 latent の patch 後の長さ {available} を超える")
        cases.append((name, patched[:, :want].contiguous()))
    return tuple(cases)


def _pristine_speaker_outputs(
    encoder: nn.Module,
    speaker_norm: nn.Module,
    cases: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """**パッチ前**の speaker 側 eager 出力（`speaker_norm` まで掛けた `[1,S,768]`）。"""
    if patch_irodori.patches_applied():
        raise AssertionError("パッチ適用後に参照を採ろうとした（同値検証が恒真化する）")
    outputs: dict[str, torch.Tensor] = {}
    for name, latent in cases:
        mask = torch.ones(latent.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            outputs[name] = speaker_norm(encoder(latent, mask))
    return outputs


def _no_reference_evidence(encoder: nn.Module, latent_dim: int, patch_size: int) -> float:
    """参照なし（`no_ref`）経路の出力が**厳密に 0** であることを実測する。

    `no_ref` は `ref_mask` を**全 0** で作る唯一の正規経路（`_load_reference_latent`）で、
    speaker グラフはマスクを持たないのでこの入力を表現できない。torch 側の出力がゼロ行列で
    あるなら、ホストは「グラフを呼ばずにゼロを置く」だけで同値になる — その主張をここで
    実測する（0 でなければ、マスクを実行時入力へ昇格させる設計判断が要る）。

    MUST: 恒真にしない — 参照 latent は**非ゼロ**を渡す。ゼロ入力だと「マスクが効いている」
    のか「入力がゼロだから 0 が出た」のか区別できない。
    """
    length = max(MIN_SYM_LENGTH, patch_size)
    generator = torch.Generator().manual_seed(999)
    latent = torch.randn(1, length, latent_dim, generator=generator)
    with torch.no_grad():
        state = encoder(latent, torch.zeros((1, length), dtype=torch.bool))
    worst = float(state.abs().max())
    if worst != 0.0:
        raise AssertionError(
            f"参照なし（マスク全 0）の speaker 出力が 0 でない（最大絶対値 {worst}）"
            " — ホスト側のゼロ供給では同値にならず、実行時マスク入力の設計が要る"
        )
    return worst


def _duration_cases(
    source: IrodoriSource,
    predictor: nn.Module,
    text_norm: nn.Module,
    caption_norm: nn.Module,
    text_proj: Mapping[str, torch.Tensor],
    caption_proj: Mapping[str, torch.Tensor],
    speaker: Mapping[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    """`duration` のグラフ入力と、**実モジュール呼び出し用**の完全な引数を組む。

    戻りは `(グラフ入力, 参照呼び出しの kwargs)`。前者は 5 本のグラフ入力そのもの、後者は
    `DurationPredictor.forward` の全引数（マスク・`aux_features` 込み）で、参照値を採るのに使う。

    MUST: speaker / caption のベクトルは実装のメソッド（`_prepend_masked_mean_token` /
    `_speaker_vec` / `_caption_vec`）を**呼んで**作る — 式を写すと上流が変わっても台本だけ
    古いまま黙って通る。
    """
    missing = [name for _n, name, _s, _c in DURATION_CASES if name not in text_proj]
    if missing:
        raise SystemExit(f"duration が参照する text ケース {missing} が golden に無い")
    for label, table, key in (
        ("speaker", speaker, DURATION_SPEAKER_SOURCE),
        ("caption-proj", caption_proj, DURATION_CAPTION_SOURCE),
    ):
        if key not in table:
            raise SystemExit(f"duration が参照する {label} ケース '{key}' が golden に無い")

    truthy = torch.ones((1,), dtype=torch.bool)
    with torch.no_grad():
        speaker_state, speaker_mask = source.prepend_masked_mean_token(
            speaker[DURATION_SPEAKER_SOURCE],
            torch.ones(speaker[DURATION_SPEAKER_SOURCE].shape[:2], dtype=torch.bool),
        )
        caption_state = caption_norm(caption_proj[DURATION_CAPTION_SOURCE])
        caption_mask = torch.ones(caption_state.shape[:2], dtype=torch.bool)
        speaker_vec = predictor._speaker_vec(
            batch_size=1,
            device=speaker_state.device,
            dtype=speaker_state.dtype,
            speaker_state=speaker_state,
            has_speaker=truthy,
        )
        caption_vec = predictor._caption_vec(
            batch_size=1,
            device=caption_state.device,
            dtype=caption_state.dtype,
            caption_state=caption_state,
            caption_mask=caption_mask,
            has_caption=truthy,
        )

    inputs: dict[str, dict[str, torch.Tensor]] = {}
    reference: dict[str, dict[str, torch.Tensor]] = {}
    for name, text_case, has_speaker, has_caption in DURATION_CASES:
        state = text_proj[text_case]
        flags = {
            "has_speaker": torch.tensor([[has_speaker]], dtype=torch.bool),
            "has_caption": torch.tensor([[has_caption]], dtype=torch.bool),
        }
        inputs[name] = {
            "text_state": state,
            # 条件が無い側は**ホストがゼロを置く**（値は where で捨てられる — グラフ内の
            # `null_*` が選ばれる）。
            "speaker_vec": speaker_vec if has_speaker else torch.zeros_like(speaker_vec),
            "has_speaker": flags["has_speaker"],
            "caption_vec": caption_vec if has_caption else torch.zeros_like(caption_vec),
            "has_caption": flags["has_caption"],
        }
        with torch.no_grad():
            normalized = text_norm(state)
        reference[name] = {
            "text_state": normalized,
            "text_mask": torch.ones(normalized.shape[:2], dtype=torch.bool),
            "speaker_state": speaker_state,
            "speaker_mask": speaker_mask,
            "has_speaker": torch.tensor([has_speaker], dtype=torch.bool),
            "caption_state": caption_state,
            "caption_mask": caption_mask,
            "has_caption": torch.tensor([has_caption], dtype=torch.bool),
        }
    return inputs, reference


def _call_duration(
    predictor: nn.Module, args: Mapping[str, torch.Tensor], aux: torch.Tensor
) -> torch.Tensor:
    """実モジュールの `DurationPredictor.forward` を全引数で呼ぶ（参照値の採取点）。"""
    with torch.no_grad():
        return predictor(
            args["text_state"],
            text_mask=args["text_mask"],
            aux_features=aux,
            speaker_state=args["speaker_state"],
            speaker_mask=args["speaker_mask"],
            has_speaker=args["has_speaker"],
            caption_state=args["caption_state"],
            caption_mask=args["caption_mask"],
            has_caption=args["has_caption"],
        ).float()


def _pristine_duration_outputs(
    predictor: nn.Module,
    reference: Mapping[str, Mapping[str, torch.Tensor]],
    aux_dim: int,
) -> dict[str, torch.Tensor]:
    """**パッチ前**の duration 出力（`[1]` の log frames）。"""
    if patch_irodori.patches_applied():
        raise AssertionError("パッチ適用後に参照を採ろうとした（同値検証が恒真化する）")
    aux = torch.zeros((1, aux_dim), dtype=torch.float32)
    return {name: _call_duration(predictor, args, aux) for name, args in reference.items()}


def _duration_aux_is_inert(
    predictor: nn.Module,
    reference: Mapping[str, Mapping[str, torch.Tensor]],
    aux_dim: int,
    pristine: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """`aux_features` が token-sum 形で**一切読まれない**ことを実測する。

    グラフから 14 次元の入力を落とす根拠がこれ 1 本なので、「読んでいないはず」を主張の
    ままにしない。別の aux（全 1）で呼び直して**ビット一致**でなければ落とす。
    """
    alternative = torch.ones((1, aux_dim), dtype=torch.float32)
    evidence: dict[str, float] = {}
    for name, args in reference.items():
        diff = float((_call_duration(predictor, args, alternative) - pristine[name]).abs().max())
        if diff != 0.0:
            raise AssertionError(
                f"{name}: aux_features を変えたら duration 出力が {diff} 動いた"
                " — token-sum 形が aux を読んでいる（グラフ入力から落とせない）"
            )
        evidence[name] = diff
    return evidence


def _right_pad(state: torch.Tensor, length: int, where: str) -> torch.Tensor:
    """`[1,T,D]` を `[1,length,D]` へ**右詰め 0 pad** する（ADR 0047 のホスト残置）。"""
    used = int(state.shape[1])
    if used > length:
        raise SystemExit(f"{where}: 長さ {used} が条件の宣言長 {length} を超えている")
    padded = state.new_zeros((int(state.shape[0]), length, int(state.shape[2])))
    padded[:, :used] = state
    return padded


def _segment_mask(length: int, used: int) -> torch.Tensor:
    """`[1,length]` の bool マスク（先頭 `used` 本が True）。`used=0` が uncond の区間。"""
    mask = torch.zeros((1, length), dtype=torch.bool)
    mask[0, :used] = True
    return mask


def _dit_cases(
    source: IrodoriSource,
    config: Any,
    model_config: Mapping[str, Any],
    text_norm: nn.Module,
    caption_norm: nn.Module,
    speaker_max: int,
    sym_max: int,
    text_proj: Mapping[str, torch.Tensor],
    caption_proj: Mapping[str, torch.Tensor],
    speaker: Mapping[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    """`dit` のグラフ入力と、**実モジュール呼び出し用**の完全な引数を組む。

    戻りは `(グラフ入力, 参照呼び出しの kwargs)`。前者は 6 本のグラフ入力そのもの、後者は
    `forward_with_encoded_conditions` の引数（**norm 済み**の条件 state と区間マスク）で、
    参照値を採るのに使う。

    MUST: 条件 state は上流ターゲットの torch 期待値から鎖にし、平均トークンの前置と
    `t_embed` は実装の関数を**呼んで**作る（式を写さない）。

    MUST: uncond 変種の参照は上流と同じ「state を 0 にしたうえでマスクも 0」で採る。
    グラフ側は cond の state のままマスクだけ 0 にするので、両者の一致が ADR 0047 決定 1 の
    実証になる（恒真ではない — `_dit_uncond_divergence` が 4 本の出力の相互差を実測する）。
    """
    for label, table, key in (
        ("text-proj", text_proj, DIT_TEXT_SOURCE),
        ("caption-proj", caption_proj, DIT_CAPTION_SOURCE),
        ("speaker", speaker, DIT_SPEAKER_SOURCE),
    ):
        if key not in table:
            raise SystemExit(f"dit が参照する {label} ケース '{key}' が golden に無い")

    with torch.no_grad():
        speaker_packed, _speaker_mask = source.prepend_masked_mean_token(
            speaker[DIT_SPEAKER_SOURCE],
            torch.ones(speaker[DIT_SPEAKER_SOURCE].shape[:2], dtype=torch.bool),
        )
    packed = {
        "text": text_proj[DIT_TEXT_SOURCE],
        # 平均トークンを前置した後の長さ（`speaker` グラフの出力 + 1）。
        "speaker": speaker_packed,
        "caption": caption_proj[DIT_CAPTION_SOURCE],
    }
    caps = {
        "text": int(model_config["max_text_len"]),
        "speaker": speaker_max + 1,
        "caption": int(model_config["max_caption_len"]),
    }
    padded = {
        name: _right_pad(state, caps[name], f"dit の {name} 条件") for name, state in packed.items()
    }
    with torch.no_grad():
        # 参照が食うのは norm 済みの state（`encode_conditions` が掛けてから DiT へ渡す）。
        # グラフは生の state を受けて同じ norm を内側で掛ける。
        normed = {
            "text": text_norm(padded["text"]),
            "speaker": padded["speaker"],
            "caption": caption_norm(padded["caption"]),
        }

    latent_dim = int(config.patched_latent_dim)
    embed_dim = int(config.timestep_embed_dim)
    inputs: dict[str, dict[str, torch.Tensor]] = {}
    reference: dict[str, dict[str, torch.Tensor]] = {}
    for name, length, seed, t_value, uncond in DIT_CASES:
        if uncond is not None and uncond not in DIT_UNCOND_VARIANTS:
            raise SystemExit(f"{name}: uncond 区間 {uncond!r} は {DIT_UNCOND_VARIANTS} に無い")
        if not MIN_SYM_LENGTH <= length <= sym_max:
            raise SystemExit(
                f"{name}: S={length} が記号次元の範囲 [{MIN_SYM_LENGTH}, {sym_max}] の外"
            )
        generator = torch.Generator().manual_seed(seed)
        x_t = torch.randn(1, length, latent_dim, generator=generator)
        t = torch.tensor([t_value], dtype=torch.float32)
        with torch.no_grad():
            t_embed = source.timestep_embedding(t, embed_dim).to(dtype=x_t.dtype)
        masks = {
            segment: _segment_mask(
                caps[segment], 0 if uncond == segment else int(packed[segment].shape[1])
            )
            for segment in DIT_UNCOND_VARIANTS
        }
        # 上流 `JointAttention.forward` の `torch.cat(context_masks, dim=1)[:, None, None, :]`
        # そのもの（self マスクは推論の全経路で全 1 — recon §1）。**並びは
        # `DIT_UNCOND_VARIANTS` の順を明示して取る** — グラフ側の K/V の連結順がこの順なので、
        # dict の挿入順に暗黙で頼ると入れ替えが静かに通る。
        mask = torch.cat(
            [
                torch.ones((1, length), dtype=torch.bool),
                *(masks[segment] for segment in DIT_UNCOND_VARIANTS),
            ],
            dim=1,
        )[:, None, None, :]
        inputs[name] = {
            "x_t": x_t,
            "t_embed": t_embed,
            "mask": mask,
            "text_state": padded["text"],
            "speaker_state": padded["speaker"],
            "caption_state": padded["caption"],
        }
        reference[name] = {
            "x_t": x_t,
            "t": t,
            **{
                f"{segment}_state": (
                    torch.zeros_like(normed[segment]) if uncond == segment else normed[segment]
                )
                for segment in DIT_UNCOND_VARIANTS
            },
            **{f"{segment}_mask": value for segment, value in masks.items()},
        }
    return inputs, reference


def _call_dit(model: nn.Module, args: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """実モジュールの `forward_with_encoded_conditions` を呼ぶ（参照値の採取点）。

    `latent_mask=None` は上流の推論経路そのまま（`JointAttention` が全 1 を作る）。
    """
    with torch.no_grad():
        return model.forward_with_encoded_conditions(
            x_t=args["x_t"],
            t=args["t"],
            text_state=args["text_state"],
            text_mask=args["text_mask"],
            speaker_state=args["speaker_state"],
            speaker_mask=args["speaker_mask"],
            caption_state=args["caption_state"],
            caption_mask=args["caption_mask"],
            latent_mask=None,
        )


def _pristine_dit_outputs(
    model: nn.Module, reference: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """**パッチ前**の DiT 出力（`[1,S,32]` の v_pred）。"""
    if patch_irodori.patches_applied():
        raise AssertionError("パッチ適用後に参照を採ろうとした（同値検証が恒真化する）")
    return {name: _call_dit(model, args) for name, args in reference.items()}


def _dit_uncond_divergence(pristine: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """マスクだけが違う 4 本（cond + uncond 3 変種）の出力が**互いに**違うことを実測する。

    MUST: 恒真にしない — 区間の割り方（self S / text / speaker / caption の並びとオフセット）が
    崩れても、ラッパと参照が同じ崩れ方をすれば golden は一致してしまう。総当たりの最小差が
    下限を割ったら落とす。
    """
    keys = {(length, seed, t) for _n, length, seed, t, uncond in DIT_CASES if uncond is not None}
    if len(keys) != 1:
        raise AssertionError(
            f"uncond 変種が x_t / t を共有していない（{sorted(keys)}）— マスクだけの差にならない"
        )
    key = keys.pop()
    group = [name for name, length, seed, t, _u in DIT_CASES if (length, seed, t) == key]
    if len(group) != 1 + len(DIT_UNCOND_VARIANTS):
        raise AssertionError(
            f"cond + uncond 3 変種の組が {group} — 比較の基準になる cond ケースが要る"
        )
    pairs = {
        f"{lhs} vs {rhs}": float((pristine[lhs] - pristine[rhs]).abs().max())
        for index, lhs in enumerate(group)
        for rhs in group[index + 1 :]
    }
    worst = min(pairs.values())
    if worst < DIT_UNCOND_DIVERGENCE_MIN:
        raise AssertionError(
            f"cond / uncond の出力差の最小が {worst} — マスクの区間割りが効いていない疑い"
        )
    return {name: round(value, 5) for name, value in pairs.items()}


def _single(outputs: Mapping[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, ...]]:
    """1 出力ターゲットの期待値を、位置つきの組（`_write_io` が食う形）へ揃える。"""
    return {name: (value,) for name, value in outputs.items()}


def _first(outputs: Mapping[str, tuple[torch.Tensor, ...]]) -> dict[str, torch.Tensor]:
    """期待値の組から**出力 0 だけ**を取り出す。

    鎖の下流（`duration` / `dit` / 静的方式の実測 / projector 取り違えの検査）が食うのは
    常に第 1 出力で、`caption-proj` の第 2 出力（`caption_norm` 済み系列）はホストが
    masked mean を採るためだけのもの。
    """
    return {name: value[0] for name, value in outputs.items()}


def _pristine_outputs(
    backbone: nn.Module,
    projectors: Mapping[str, nn.Module],
    caption_norm: nn.Module,
    cases: Sequence[tuple[str, str, torch.Tensor]],
) -> dict[str, dict[str, tuple[torch.Tensor, ...]]]:
    """**パッチ前**の eager 出力（golden の期待値そのもの）をターゲット別に採る。

    MUST: `patch_irodori` を当てる前に呼ぶ（当てた後だと同値検証が恒真化する）。
    実モジュールの forward をそのまま通す（全 1 マスク）— ラッパの写し間違いは
    `_check_wrapper_equivalence` がここで採った値との差として出る。

    `caption-proj` だけ期待値が 2 本になる（第 2 出力 = `caption_norm` 適用済み系列）。
    """
    if patch_irodori.patches_applied():
        raise AssertionError("パッチ適用後に参照を採ろうとした（同値検証が恒真化する）")
    outputs: dict[str, dict[str, tuple[torch.Tensor, ...]]] = {
        target: {} for target in TEXT_TARGETS
    }
    for name, _kind, ids in cases:
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            outputs[TARGET_BACKBONE][name] = (backbone(ids, mask),)
            for target, projector in projectors.items():
                projected = projector(backbone, ids, mask)
                outputs[target][name] = (
                    (projected, caption_norm(projected))
                    if target == TARGET_CAPTION_PROJ
                    else (projected,)
                )
    return outputs


def _static_scheme_evidence(
    backbone: nn.Module,
    text_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    cases: Sequence[tuple[str, str, torch.Tensor]],
    pristine: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """「右詰め pad + マスク」と「詰めた列」が同値であることを実測する（静的方式の根拠）。

    Irodori の推論は固定長（text 256 / caption 512）へ右詰め pad して backbone を呼ぶ。
    静的方式はその pad を**ホスト側で消す**ので、この 2 つが一致しなければ方式そのものが
    成立しない。**回避ではなく実測で決着させる**ため、ここは常に走る門にしてある。
    """
    pad_id = int(text_config["pad_token_id"])
    caps = {
        "text": int(model_config["max_text_len"]),
        "caption": int(model_config["max_caption_len"]),
    }
    evidence: dict[str, float] = {}
    for name, kind, ids in cases:
        length = int(ids.shape[1])
        padded_len = caps[kind]
        if length > padded_len:
            raise SystemExit(f"{name}: T={length} が {kind} の上限 {padded_len} を超えている")
        padded_ids = torch.full((1, padded_len), pad_id, dtype=torch.int64)
        padded_ids[0, :length] = ids[0]
        padded_mask = torch.zeros((1, padded_len), dtype=torch.bool)
        padded_mask[0, :length] = True
        with torch.no_grad():
            padded_state = backbone(padded_ids, padded_mask)
        diff = float((padded_state[:, :length] - pristine[name]).abs().max())
        if diff > STATIC_SCHEME_ATOL:
            raise AssertionError(
                f"{name}: pad 付き呼び出しの先頭 {length} 行と詰めた列の差 {diff} が"
                f" {STATIC_SCHEME_ATOL} を超えた — 静的方式（実行時 attention_mask 非対応）が"
                "このモデルでは成立しない"
            )
        evidence[name] = diff
    return evidence


def _check_wrapper_equivalence(
    wrapper: nn.Module,
    args: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
    where: str,
    atol: float,
) -> float:
    """ラッパの出力がパッチ前の実モジュール出力と一致することを見る（多出力対応）。"""
    with torch.no_grad():
        actual = wrapper(*args)
    outputs = actual if isinstance(actual, tuple) else (actual,)
    if len(outputs) != len(expected):
        raise AssertionError(
            f"{where}: ラッパの出力が {len(outputs)} 本で期待値 {len(expected)} 本と違う"
        )
    diff = max(
        float((value - want).abs().max()) for value, want in zip(outputs, expected, strict=True)
    )
    if diff > atol:
        raise AssertionError(f"{where}: eager 同値が崩れた（最大絶対差 {diff} > {atol}）")
    return diff


def _write_io(
    graph: IrGraph,
    inputs: Mapping[str, Mapping[str, torch.Tensor]],
    expected: Mapping[str, Sequence[torch.Tensor]],
    out_dir: Path,
) -> list[str]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    期待値は**位置つきの組**で受ける（`caption-proj` だけ 2 本 — モジュール docstring の
    G1b 節）。本数が IR の出力数と食い違う io は「読めるが 1 本ぶん検証されない」形で
    静かに通るので、ケースごとに突き合わせる。
    """
    declared = [spec.name for spec in graph.inputs]
    written: list[str] = []
    for name, args in inputs.items():
        if sorted(args) != sorted(declared):
            raise AssertionError(
                f"{name}: 入力名 {sorted(args)} が IR の {sorted(declared)} と違う"
            )
        outputs = expected[name]
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{name}: 期待出力 {len(outputs)} 本が IR 出力 {len(graph.outputs)} 本と違う"
            )
        # MUST: io も IR の意味論 dtype の実表現へ落とす（i64 → i32）。ランタイムが受け取る
        # 形と揃っていないと Deno 側 E2E が golden を読めない（ADR 0009 の境界正規化）。
        tensors = {
            f"{INPUT_PREFIX}{key}": normalize_boundary_tensor(value, f"{name} の入力 '{key}'")
            for key, value in args.items()
        }
        for index, value in enumerate(outputs):
            tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
                value.detach().contiguous(), f"{name} の出力 {index}"
            )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def _sanity(pristine: Mapping[str, Mapping[str, Sequence[torch.Tensor]]]) -> dict[str, Any]:
    """text 側と caption 側の projector が**別物**であることを見る（比べるのは第 1 出力）。

    MUST: 恒真にしない — 同じ接頭辞から 2 回読む取り違えは、shape も dtype も一致するので
    ここ以外では見えない（golden も両方同じ値で焼かれて一致してしまう）。
    """
    divergence = {
        name: float(
            (pristine[TARGET_TEXT_PROJ][name][0] - pristine[TARGET_CAPTION_PROJ][name][0])
            .abs()
            .max()
        )
        for name in pristine[TARGET_TEXT_PROJ]
    }
    worst = min(divergence.values())
    if worst < PROJECTOR_DIVERGENCE_MIN:
        raise AssertionError(
            f"text / caption projector の出力差の最小が {worst} — 同じ重みを 2 回読んでいる疑い"
        )
    return {name: round(value, 5) for name, value in divergence.items()}


def _norm_divergence(text_norm: nn.Module, caption_norm: nn.Module) -> float:
    """`text_norm` と `caption_norm` が**別物**であることを見る（weight の最大絶対差）。

    MUST: 恒真にしない — どちらも `[512]` の RMSNorm weight なので、同じ接頭辞から 2 回
    読む取り違えは shape も dtype も一致する。`caption-proj` の第 2 出力は
    「`caption_norm` を掛けた系列」であることが契約の全てなのに、**参照もラッパも同じ
    モジュールを使う**ので、取り違えても golden は自己一致して静かに通る（`_sanity` が
    projector について塞いでいる穴の norm 版）。
    """
    worst = float((text_norm.weight - caption_norm.weight).detach().abs().max())
    if worst < NORM_DIVERGENCE_MIN:
        raise AssertionError(
            f"text_norm / caption_norm の weight 差が {worst} — 同じ重みを 2 回読んでいる疑い"
        )
    return worst


class TargetAxis(NamedTuple):
    """ターゲット別の記号次元の宣言と、SDPA の扱い。

    - `torch_dim` / `symbol`: torch の `Dim` 名と IR に載る名前。**別々に持つ** —
      `dit` は torch 名に `"S"` を使えない（`DIT_TORCH_DIM` の MUST）。
    - `dynamic`: **入力位置 → (軸, オフセット)**。オフセット 0 は素のシンボル、非 0 は
      派生次元（`S+1519` — ADR 0046 / `dit` の `mask` だけ）。
    - `preserved`: 分解表から外す op。融合 attention（ADR 0023）を使うのはマスクが無いか
      加算型 `[1,1,M,N]` のターゲットだけで、`dit` は実行時 bool マスクなので分解経路
      （+ `safe_softmax` — ADR 0044）に落とす。
    """

    torch_dim: str
    symbol: str
    upper: int
    dynamic: Mapping[int, tuple[int, int]]
    preserved: tuple[str, ...]


def _dynamic_axis(found: tuple[int, int] | None, seq: Any) -> dict[int, Any] | None:
    """{@link TargetAxis.dynamic} の 1 要素を `torch.export` の `dynamic_shapes` へ落とす。"""
    if found is None:
        return None
    axis, offset = found
    return {axis: seq if offset == 0 else seq + offset}


def _graph_summary(graph: IrGraph, path: Path) -> dict[str, Any]:
    return {
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "model_bytes": path.stat().st_size,
    }


class FakeQuantResult(NamedTuple):
    """{@link fake_quant} の戻り（丸めの要約と、i8 の per-channel scale 台帳）。"""

    #: 役割名 → 丸めた本数の要約（`--dtype` を渡したのに 0 本、を沈黙させないための計数）。
    reports: dict[str, str]
    #: 役割名 → **素のモジュール内 FQN** → scale（i8 以外は空）。emit が引く**ラッパ内 FQN**
    #: への張り替えは {@link target_scales} が行う。
    scales: dict[str, Mapping[str, torch.Tensor]]


#: ターゲット → そのグラフに載せる scale の出どころ `(役割, 素の FQN 接頭辞,
#: ラッパ内 FQN 接頭辞)` の並び（i8 のみ使う）。
#:
#: MUST: グラフラッパのコンストラクタと同じ綴りにする — ラッパは `load_*` が返した
#: モジュールを別の属性名で抱えるので、台帳のキー（素のモジュール内 FQN）は emit が引く
#: キー（export したモジュール内 FQN）と一致しない。食い違えば `target_scales` か emit の
#: どちらかが fail loudly で落ちる（黙って f32 格納に落ちる経路は無い）。
TARGET_SCALE_SOURCES: Mapping[str, tuple[tuple[str, str, str], ...]] = MappingProxyType(
    {
        # `BackboneGraph` は `backbone.backbone`（ModernBERT 本体）を `model` に持つ。
        TARGET_BACKBONE: ((TARGET_BACKBONE, "backbone.", "model."),),
        TARGET_TEXT_PROJ: ((TARGET_TEXT_PROJ, "", "projector."),),
        # `CaptionProjectorGraph` は `ProjectorGraph` を `projection` に内包する（2 段）。
        TARGET_CAPTION_PROJ: ((TARGET_CAPTION_PROJ, "", "projection.projector."),),
        TARGET_SPEAKER: ((TARGET_SPEAKER, "", "encoder."),),
        TARGET_DURATION: ((TARGET_DURATION, "", "predictor."),),
        # `DitGraph` は DiT 本体の部分木を**同じ属性名**で持つ（張り替え不要）。台帳には
        # `load_dit` が丸ごと組む内側のコピー（backbone 等）も載るが、ラッパの
        # `named_parameters` に無いので落ちる。
        TARGET_DIT: ((TARGET_DIT, "", ""),),
    }
)


def _has_quantizable_weights(module: nn.Module) -> bool:
    """per-channel i8 の対象型（`QUANT_CHANNEL_AXES`）を 1 本でも含むか。"""
    types = tuple(QUANT_CHANNEL_AXES)
    return any(isinstance(child, types) for child in module.modules())


def target_scales(
    target: str, wrapper: nn.Module, scales: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """役割ごとの scale 台帳を、そのターゲットの**ラッパ内 FQN**へ張り替えて 1 本に束ねる。

    emit は initializer のテンソルキー（= export したモジュール内 FQN）で scale を引く
    （`karume.emit._plan_i8`）ので、`load_*` が返す素のモジュール基準の台帳をそのまま
    渡すと 1 本も当たらない。張り替え表は {@link TARGET_SCALE_SOURCES}。

    ここで落とすのは**ラッパに無い重み**だけ（`dit` の台帳は使わない内側のコピーまで含む）。
    逆向き（ラッパにあるのに台帳に無い）は emit が fail loudly で落とす — 適格な重みスロットに
    scale が無ければ i8 格納にできないため、黙って f32 へ落ちる経路は無い。
    """
    if not scales:
        return {}
    owned = {name for name, _ in wrapper.named_parameters()}
    picked: dict[str, torch.Tensor] = {}
    for role, source, destination in TARGET_SCALE_SOURCES[target]:
        for key, scale in scales.get(role, {}).items():
            if not key.startswith(source):
                continue
            rebased = destination + key[len(source) :]
            if rebased in owned:
                picked[rebased] = scale
    if not picked:
        raise SystemExit(
            f"{target}: scale 台帳が 1 本もラッパ内 FQN へ張り替えられなかった"
            f"（{TARGET_SCALE_SOURCES[target]} の接頭辞がラッパの構成と食い違っている）"
        )
    return picked


def fake_quant(dtype: str, modules: Mapping[str, nn.Module]) -> FakeQuantResult:
    """格納 dtype の表現可能値へ**実効重み**を丸める（f32 は何もしない）。

    ADR 0006 / 0018 / 0019 / 0027 / 0050 の fake-quant。丸めた本数を役割名で引ける形にして
    返す（`--dtype f16` を渡したのに 0 本、を沈黙させないため — 総数 0 は落とす）。

    MUST（順序）: ① `load_*` の**直後**に呼ぶ。この台本の重みは `convert_dacvae.py` /
    チェックポイントの時点で実効重み（`remove_weight_norm` 相当は変換時に焼き込み済み・
    LoRA も無い）なので、SBV2 の「remove の後」に相当する時点が load 直後まで前倒しになる。
    ② 参照・golden の採取の**前**に呼ぶ（`karume.quantize` の MUST — 後に当てると golden
    だけが元の重みで計算され、E2E の差に量子化誤差が混ざって tolerance の意味が消える）。

    MUST: 束ねる相手は**グラフに載る 6 モジュール + ホストが golden 入力を組むのに使う
    3 つの norm**の全部。`text_norm` / `caption_norm` は `duration` / `dit` の内側にも
    別コピーで居るが、両方を同じ値へ丸めるので鎖（上流ターゲットの期待値を次のターゲットの
    入力に使う形）は保たれる。1 つでも漏らすと、漏れた側だけが元の重みで計算した値を
    golden に載せる。

    NOTE: f16 で丸めるのは f32 のパラメータ / バッファだけ（`round_weights_to_f16`）。グラフ
    定数（帯マスク・RoPE 表・`sym_prefix_slice` の添字表）はモジュールの重みではないので触らず、
    emit 側の適格判定でも f32 格納のまま残る。

    NOTE: i8 が触るのは `QUANT_CHANNEL_AXES` の型（Linear / Conv / Embedding）の `weight` だけ
    （ADR 0019）。bias も norm 系の weight も**丸めない** — emit の適格判定でも重みスロット
    以外は f32 格納なので、両者は整合する。したがって 3 つの norm は i8 では素通りが正しく、
    「対象 0 本」を役割単位では落とさない（**全体で** 0 本なら落とす）。
    """
    if dtype == "f32":
        return FakeQuantResult({}, {})
    reports: dict[str, str] = {}
    scales: dict[str, Mapping[str, torch.Tensor]] = {}
    for name, module in sorted(modules.items()):
        if dtype != "i8":
            reports[name] = f"格納 f16 へ丸めた — {round_weights_to_f16(module).describe()}"
        elif _has_quantizable_weights(module):
            int8 = fake_quant_int8(module)
            scales[name] = int8.scales
            reports[name] = f"格納 i8 へ丸めた — {int8.describe()}"
        else:
            reports[name] = "格納 f32 のまま（per-channel の対象型を持たない）"
    for name, report in reports.items():
        print(f"[fake-quant] {name}: {report}", flush=True)
    if not reports:
        raise SystemExit(f"格納 {dtype} を指定したが丸める対象のモジュールが 1 本も無い")
    if dtype == "i8" and not scales:
        raise SystemExit("格納 i8 を指定したが per-channel 量子化できたモジュールが 1 本も無い")
    return FakeQuantResult(reports, scales)


def export_series(
    model_dir: Path,
    source_dir: Path,
    out_dir: Path,
    *,
    targets: Sequence[str] = TARGETS,
    sym_max: int = SYM_MAX,
    dtype: str = "f32",
) -> dict[str, Any]:
    """IR コンテナと golden io を書き、要約を返す。"""
    source = IrodoriSource(source_dir)
    text_config, model_config = read_configs(model_dir)
    if str(model_config["pretrained_projector_type"]) != "residual_mlp":
        raise SystemExit(
            f"projector 型 {model_config['pretrained_projector_type']!r} は未対応"
            "（v4-Small は residual_mlp）"
        )
    state = load_file(str(model_dir / MODEL_FILE))
    backbone = load_backbone(source, state, text_config)
    hidden_size = int(backbone.hidden_size)
    projectors = {
        TARGET_TEXT_PROJ: load_projector(
            source,
            state,
            model_config,
            TEXT_PROJ_PREFIX,
            hidden_size,
            int(model_config["text_dim"]),
        ),
        TARGET_CAPTION_PROJ: load_projector(
            source,
            state,
            model_config,
            CAPTION_PROJ_PREFIX,
            hidden_size,
            int(model_config["caption_dim"]),
        ),
    }
    config = source.model_config(model_config)
    speaker_encoder = load_speaker_encoder(source, state, config)
    speaker_norm = load_rms_norm(
        source, state, SPEAKER_NORM_PREFIX, int(config.speaker_dim), float(config.norm_eps)
    )
    text_norm = load_rms_norm(
        source, state, TEXT_NORM_PREFIX, int(config.text_dim), float(config.norm_eps)
    )
    caption_norm = load_rms_norm(
        source,
        state,
        CAPTION_NORM_PREFIX,
        int(config.caption_dim_resolved),
        float(config.norm_eps),
    )
    duration = load_duration_predictor(source, state, config)
    dit = load_dit(source, state, config, text_config)
    # MUST: 丸めは load の直後・参照の採取より前（{@link fake_quant} の順序 MUST）。
    quantized = fake_quant(
        dtype,
        {
            TARGET_BACKBONE: backbone,
            TARGET_TEXT_PROJ: projectors[TARGET_TEXT_PROJ],
            TARGET_CAPTION_PROJ: projectors[TARGET_CAPTION_PROJ],
            TARGET_SPEAKER: speaker_encoder,
            TARGET_DURATION: duration,
            TARGET_DIT: dit,
            "speaker_norm": speaker_norm,
            "text_norm": text_norm,
            "caption_norm": caption_norm,
        },
    )
    cases = build_cases(model_dir, source, text_config, sym_max)
    speaker_max = speaker_sym_max(model_config)
    speaker_cases = (
        *build_speaker_cases(int(config.speaker_patched_latent_dim), speaker_max),
        *build_real_speaker_cases(
            source.patch_sequence_with_mask,
            int(config.latent_dim),
            int(config.speaker_patch_size),
            int(config.speaker_patched_latent_dim),
            speaker_max,
        ),
    )
    dit_max = dit_sym_max(config)
    #: `dit` の mask の条件側の総長（派生次元 `S+<これ>` — ADR 0047 の 1519）。
    dit_context_total = (
        int(model_config["max_text_len"]) + (speaker_max + 1) + int(model_config["max_caption_len"])
    )

    # ---- ① パッチ前の参照（golden の期待値）と方式の実測 ----
    pristine = _pristine_outputs(backbone, projectors, caption_norm, cases)
    static_evidence = _static_scheme_evidence(
        backbone, text_config, model_config, cases, _first(pristine[TARGET_BACKBONE])
    )
    sanity = _sanity(pristine)
    norm_divergence = _norm_divergence(text_norm, caption_norm)
    pristine[TARGET_SPEAKER] = _single(
        _pristine_speaker_outputs(speaker_encoder, speaker_norm, speaker_cases)
    )
    no_reference_max_abs = _no_reference_evidence(
        speaker_encoder,
        int(config.speaker_patched_latent_dim),
        int(config.speaker_patch_size),
    )
    duration_inputs, duration_reference = _duration_cases(
        source,
        duration,
        text_norm,
        caption_norm,
        _first(pristine[TARGET_TEXT_PROJ]),
        _first(pristine[TARGET_CAPTION_PROJ]),
        _first(pristine[TARGET_SPEAKER]),
    )
    aux_dim = int(config.duration_aux_dim)
    duration_pristine = _pristine_duration_outputs(duration, duration_reference, aux_dim)
    pristine[TARGET_DURATION] = _single(duration_pristine)
    aux_inert = _duration_aux_is_inert(duration, duration_reference, aux_dim, duration_pristine)
    dit_inputs, dit_reference = _dit_cases(
        source,
        config,
        model_config,
        text_norm,
        caption_norm,
        speaker_max,
        dit_max,
        _first(pristine[TARGET_TEXT_PROJ]),
        _first(pristine[TARGET_CAPTION_PROJ]),
        _first(pristine[TARGET_SPEAKER]),
    )
    dit_pristine = _pristine_dit_outputs(dit, dit_reference)
    pristine[TARGET_DIT] = _single(dit_pristine)
    dit_divergence = _dit_uncond_divergence(dit_pristine)

    # ---- ② パッチ適用（ここから先で採った eager 値は参照に使えない） ----
    rope_buffers = sorted(
        name for name, _ in backbone.backbone.named_buffers() if name.endswith(ROPE_BUFFER_NAMES)
    )
    patch_irodori.apply_patches()

    # ---- ③ ラッパの eager 同値（パッチ + マスク落としの両方をここで実測する） ----
    graphs = {
        TARGET_BACKBONE: BackboneGraph(backbone),
        TARGET_TEXT_PROJ: ProjectorGraph(projectors[TARGET_TEXT_PROJ]),
        TARGET_CAPTION_PROJ: CaptionProjectorGraph(projectors[TARGET_CAPTION_PROJ], caption_norm),
        TARGET_SPEAKER: SpeakerGraph(speaker_encoder, speaker_norm, speaker_max),
        TARGET_DURATION: DurationGraph(duration, text_norm),
        TARGET_DIT: DitGraph(dit, dit_max),
    }
    graph_args: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        TARGET_BACKBONE: {name: {"input_ids": ids} for name, _kind, ids in cases},
        TARGET_SPEAKER: {name: {"latent": latent} for name, latent in speaker_cases},
        TARGET_DURATION: duration_inputs,
        TARGET_DIT: dit_inputs,
    }
    for target in (TARGET_TEXT_PROJ, TARGET_CAPTION_PROJ):
        graph_args[target] = {
            name: {"hidden": pristine[TARGET_BACKBONE][name][0]} for name, _kind, _ids in cases
        }
    equivalence = {
        target: max(
            _check_wrapper_equivalence(
                module,
                tuple(args.values()),
                pristine[target][name],
                f"{target}/{name}",
                EAGER_EQUIV_ATOL,
            )
            for name, args in graph_args[target].items()
        )
        for target, module in graphs.items()
    }

    # ---- ④ export と golden の書き出し ----
    # ターゲット別の記号次元の宣言（{@link TargetAxis}）。テキスト系 3 本と duration は
    # T（512）で、speaker は S（参照 latent の patch 後上限）、dit も S（latent 長 750 —
    # 別グラフなので名前は衝突しない）。duration の 2〜5 本目（条件ベクトルと bool）と
    # dit の 2 / 4〜6 本目（t_embed と条件 state）は記号を持たない固定 shape。
    axes: dict[str, TargetAxis] = {
        TARGET_BACKBONE: TargetAxis(
            TEXT_SYMBOL, TEXT_SYMBOL, sym_max, {0: (1, 0)}, PRESERVED_OP_PREFIXES_WITH_ATTENTION
        ),
        TARGET_TEXT_PROJ: TargetAxis(
            TEXT_SYMBOL, TEXT_SYMBOL, sym_max, {0: (1, 0)}, PRESERVED_OP_PREFIXES_WITH_ATTENTION
        ),
        TARGET_CAPTION_PROJ: TargetAxis(
            TEXT_SYMBOL, TEXT_SYMBOL, sym_max, {0: (1, 0)}, PRESERVED_OP_PREFIXES_WITH_ATTENTION
        ),
        TARGET_SPEAKER: TargetAxis(
            SPEAKER_SYMBOL,
            SPEAKER_SYMBOL,
            speaker_max,
            {0: (1, 0)},
            PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        ),
        TARGET_DURATION: TargetAxis(
            TEXT_SYMBOL, TEXT_SYMBOL, sym_max, {0: (1, 0)}, PRESERVED_OP_PREFIXES_WITH_ATTENTION
        ),
        TARGET_DIT: TargetAxis(
            DIT_TORCH_DIM,
            DIT_SYMBOL,
            dit_max,
            {0: (1, 0), 2: (3, dit_context_total)},
            PRESERVED_OP_PREFIXES,
        ),
    }
    written: dict[str, Any] = {}
    for target in targets:
        axis = axes[target]
        target_dir = out_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        by_case = graph_args[target]
        longest = max(by_case, key=lambda name: next(iter(by_case[name].values())).shape[1])
        example = tuple(by_case[longest].values())
        seq = Dim(axis.torch_dim, min=MIN_SYM_LENGTH, max=axis.upper)
        graph = export_to_file(
            graphs[target],
            example,
            target_dir / MODEL_FILE,
            dynamic_shapes=tuple(
                _dynamic_axis(axis.dynamic.get(index), seq) for index in range(len(example))
            ),
            symbol_names=(axis.symbol,),
            preserved=axis.preserved,
            weight_dtype=dtype,
            weight_scales=target_scales(target, graphs[target], quantized.scales),
        )
        io_files = _write_io(graph, by_case, pristine[target], target_dir)
        written[target] = {
            **_graph_summary(graph, target_dir / MODEL_FILE),
            "io": io_files,
        }
    return {
        "dir": str(out_dir),
        "dtype": dtype,
        "fake_quant": quantized.reports,
        "targets": written,
        "case_lengths": {name: int(ids.shape[1]) for name, _kind, ids in cases},
        "speaker_case_lengths": {name: int(latent.shape[1]) for name, latent in speaker_cases},
        "speaker_sym_max": speaker_max,
        "dit_case_lengths": {name: int(args["x_t"].shape[1]) for name, args in dit_inputs.items()},
        "dit_sym_max": dit_max,
        "dit_context_total": dit_context_total,
        "dit_uncond_divergence": dit_divergence,
        "rope_buffers_lifted": rope_buffers,
        "static_scheme_max_abs": {k: float(f"{v:.3e}") for k, v in static_evidence.items()},
        "no_reference_max_abs": no_reference_max_abs,
        "duration_aux_inert_max_abs": aux_inert,
        "eager_equivalence_max_abs": equivalence,
        "projector_divergence": sanity,
        "norm_divergence": round(norm_divergence, 5),
    }


def default_out_root(model_dir: Path, dtype: str = "f32") -> Path:
    """生成物の既定の置き場（`outputs/series/irodori-<実重みのディレクトリ名>{,-f16,-i8}/`）。

    ターゲット名のサブディレクトリは `export_series` が 1 段掘る（`export_sbv2.py` と同じ形）。

    MUST: dtype ごとに別ディレクトリ（ADR 0018 / 0019 / 0027）— 同居させると f32 系列の網（実測から
    導いたターゲット別 tolerance）が圧縮資産へ黙って掛かる。綴りは `karume.dist` の
    `irodori_series_name` + dtype 接尾と一致させる（書き手と読み手が同じ 1 語から組む）。
    """
    suffix = "" if dtype == "f32" else f"-{dtype}"
    return SERIES_ROOT / f"irodori-{model_dir.name}{suffix}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は --dtype ごとの系列 —"
        " outputs/series/irodori-<--model-dir のディレクトリ名>{,-f16,-i8}/）",
    )
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help="重みの格納 dtype（f16 / i8 は fake-quant してから適格スロットだけ圧縮格納する"
        " — ADR 0018 / 0019 / 0027 / 0050。**emit 専用**）",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        help="書き出すターゲット（繰り返し指定可。既定は全て）",
    )
    args = parser.parse_args(argv)
    out_dir = default_out_root(args.model_dir, args.dtype) if args.out is None else args.out
    summary = export_series(
        args.model_dir,
        args.source_dir,
        out_dir,
        targets=tuple(args.target) if args.target else TARGETS,
        sym_max=args.sym_max,
        dtype=args.dtype,
    )
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
