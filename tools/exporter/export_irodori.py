"""実重み Irodori-TTS v4-Small を IR v1 コンテナ + golden io へ書き出す台本。

今回のスコープは**テキスト条件エンコーダ**（recon の G1 / G1a / G1b）と **speaker encoder /
duration predictor**（同 G2 / G3）で、DiT や codec（G4〜G7）は後続の波でこの台本に
ターゲットとして足す。

    cd tools/exporter
    uv run --with 'transformers==5.14.1' python export_irodori.py
    uv run --with 'transformers==5.14.1' python export_irodori.py --target backbone

transformers は **5.14.1 でピン**する（`export_embeddinggemma.py` と同じ理由 — モデリング
コードが変わるとグラフ形が変わる。加えて `karume.patch_irodori` が
`ModernBertAttention.forward` をクラス属性ごと差し替える）。pyproject.toml / uv.lock には
入れず `--with` で一時的に足す。

モデル実装は GitHub `Aratako/Irodori-TTS` の clone（既定 `inputs/irodori/Irodori-TTS/`）から
**`sys.path` 追加で import** する（`--source-dir`）。`irodori_tts.model` の import は
`irodori_tts/__init__.py` 経由になるが、そこが引く追加依存は transformers だけなので、
限定 import の細工は要らない。

## 何をグラフに載せるか（5 ターゲット・B=1・T / S は記号次元）

| ターゲット     | 入力          | 出力        | 中身                                  |
| -------------- | ------------- | ----------- | ------------------------------------- |
| `backbone`     | `[1,T]` ids   | `[1,T,768]` | ModernBERT-ja-310m（25 層・共有）     |
| `text-proj`    | `[1,T,768]`   | `[1,T,512]` | text 側 projector（residual_mlp）     |
| `caption-proj` | `[1,T,768]`   | `[1,T,512]` | caption 側 projector（同形・別重み）  |
| `speaker`      | `[1,S,128]`   | `[1,S,768]` | `ReferenceLatentEncoder` + 出力 norm  |
| `duration`     | 下記 5 本     | `[1]`       | `text_norm` + duration（token-sum 形）|

`duration` の入力 5 本は `text_state [1,T,512]` / `speaker_vec [1,768]` /
`has_speaker [1,1]`（bool）/ `caption_vec [1,512]` / `has_caption [1,1]`（bool）。

backbone を projector と融合しないのは、**backbone が text と caption で共有**だから
（融合すると 1.2GB の重みが 2 系列に複製される）。ホストは backbone を 2 回回して、
それぞれの projector へ流す。

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
— `patch_irodori` のモジュール docstring）。RMSNorm 分割はビット一致が構造的に成り立ち、
RoPE の実数化は**この重みの幾何（head_dim 64）で**ビット一致することが実測なので、
`EAGER_EQUIV_ATOL = 0` の同値検証はテキスト系 3 本と同じ厳しさのまま通る（head_dim が
変われば 1 ulp ずれうる — その時は 0 のまま落ちるので、緩める前に幾何を疑う）。

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
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.export import Dim

from karume import patch_irodori
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph
from karume.patch_anima import ROPE_BUFFER_NAMES, assert_rope_lifted
from karume.paths import INPUTS_ROOT, SERIES_ROOT
from karume.pipeline import export_to_file

#: 実重みの置き場（`hf download Aratako/Irodori-TTS-v4-Small` の展開先）。
DEFAULT_MODEL_DIR = INPUTS_ROOT / "irodori" / "v4-small"

#: モデル実装（GitHub `Aratako/Irodori-TTS` の clone）の置き場。
DEFAULT_SOURCE_DIR = INPUTS_ROOT / "irodori" / "Irodori-TTS"

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
#: 同じ token 列 golden ケースを共有する 3 本（backbone を 2 つの projector が食う鎖）。
TEXT_TARGETS = (TARGET_BACKBONE, TARGET_TEXT_PROJ, TARGET_CAPTION_PROJ)
TARGETS = (*TEXT_TARGETS, TARGET_SPEAKER, TARGET_DURATION)

#: テキスト系 3 本の記号次元名（IR に載る名前）。
TEXT_SYMBOL = "T"

#: `speaker` の記号次元名。参照 latent の**patch 後**の長さで、テキストの T とは別の軸。
SPEAKER_SYMBOL = "S"

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

#: `speaker` の golden ケース `(名前, patch 後トークン数 S, 乱数 seed)`。
#:
#: 参照 latent は**合成**（決定的 seed の `torch.randn`）で良い — この門が見ているのは
#: 「karume の実行が torch と一致するか」であって、実音声の再現ではない（実音声から
#: latent を採るには別リポの DACVAE 重みが要る。それを門の前提にすると、コーデック波が
#: 済むまで speaker の門が立たない）。
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
        )
        from irodori_tts.text_normalization import normalize_text

        self.backbone_cls = PretrainedTextBackbone
        self.projector_cls = PretrainedConditionProjector
        self.speaker_cls = ReferenceLatentEncoder
        self.duration_cls = DurationPredictor
        self.rms_norm_cls = RMSNorm
        self.model_config_cls = ModelConfig
        #: 平均トークンの前置は `TextToLatentRFDiT` の staticmethod。台本は式を写さずに呼ぶ。
        self.prepend_masked_mean_token = TextToLatentRFDiT._prepend_masked_mean_token
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


def build_cases(
    model_dir: Path, source: IrodoriSource, text_config: Mapping[str, Any], sym_max: int
) -> tuple[tuple[str, str, torch.Tensor], ...]:
    """golden ケースの `(名前, 種別, input_ids)`。

    前処理は実装の正本（`inference_runtime._synthesize`）と同じ順序:
    正規化（`normalize_text` + `strip`）→ 特殊トークン無しでトークナイズ → BOS 前置。
    右詰め pad は**しない**（静的方式ではホストが列を詰める — モジュール docstring）。
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
    bos_id = int(text_config["bos_token_id"])
    cases: list[tuple[str, str, torch.Tensor]] = []
    for name, kind, body in GOLDEN_CASES:
        normalized = source.normalize_text(body).strip()
        if not normalized:
            raise SystemExit(f"{name}: 正規化後の本文が空")
        ids = [bos_id, *tokenizer.encode(normalized, add_special_tokens=False).ids]
        if not 2 <= len(ids) <= sym_max:
            raise SystemExit(f"{name}: T={len(ids)} が記号次元の範囲 [2, {sym_max}] の外")
        cases.append((name, kind, torch.tensor([ids], dtype=torch.int64)))
    return tuple(cases)


def build_speaker_cases(latent_dim: int, sym_max: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """`speaker` の golden ケース `(名前, 合成参照 latent)`。

    値は決定的 seed の標準正規（`SPEAKER_CASES` の注記 — 実音声 latent は別リポの重みが要る）。
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


def _pristine_outputs(
    backbone: nn.Module,
    projectors: Mapping[str, nn.Module],
    cases: Sequence[tuple[str, str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """**パッチ前**の eager 出力（golden の期待値そのもの）をターゲット別に採る。

    MUST: `patch_irodori` を当てる前に呼ぶ（当てた後だと同値検証が恒真化する）。
    実モジュールの forward をそのまま通す（全 1 マスク）— ラッパの写し間違いは
    `_check_wrapper_equivalence` がここで採った値との差として出る。
    """
    if patch_irodori.patches_applied():
        raise AssertionError("パッチ適用後に参照を採ろうとした（同値検証が恒真化する）")
    outputs: dict[str, dict[str, torch.Tensor]] = {target: {} for target in TEXT_TARGETS}
    for name, _kind, ids in cases:
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            outputs[TARGET_BACKBONE][name] = backbone(ids, mask)
            for target, projector in projectors.items():
                outputs[target][name] = projector(backbone, ids, mask)
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
    expected: torch.Tensor,
    where: str,
    atol: float,
) -> float:
    """ラッパの出力がパッチ前の実モジュール出力と一致することを見る。"""
    with torch.no_grad():
        actual = wrapper(*args)
    diff = float((actual - expected).abs().max())
    if diff > atol:
        raise AssertionError(f"{where}: eager 同値が崩れた（最大絶対差 {diff} > {atol}）")
    return diff


def _write_io(
    graph: IrGraph,
    inputs: Mapping[str, Mapping[str, torch.Tensor]],
    expected: Mapping[str, torch.Tensor],
    out_dir: Path,
) -> list[str]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。"""
    if len(graph.outputs) != 1:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（テキスト系は 1 本）")
    declared = [spec.name for spec in graph.inputs]
    written: list[str] = []
    for name, args in inputs.items():
        if sorted(args) != sorted(declared):
            raise AssertionError(
                f"{name}: 入力名 {sorted(args)} が IR の {sorted(declared)} と違う"
            )
        # MUST: io も IR の意味論 dtype の実表現へ落とす（i64 → i32）。ランタイムが受け取る
        # 形と揃っていないと Deno 側 E2E が golden を読めない（ADR 0009 の境界正規化）。
        tensors = {
            f"{INPUT_PREFIX}{key}": normalize_boundary_tensor(value, f"{name} の入力 '{key}'")
            for key, value in args.items()
        }
        tensors[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
            expected[name].detach().contiguous(), f"{name} の出力 0"
        )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def _sanity(pristine: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    """text 側と caption 側の projector が**別物**であることを見る。

    MUST: 恒真にしない — 同じ接頭辞から 2 回読む取り違えは、shape も dtype も一致するので
    ここ以外では見えない（golden も両方同じ値で焼かれて一致してしまう）。
    """
    divergence = {
        name: float(
            (pristine[TARGET_TEXT_PROJ][name] - pristine[TARGET_CAPTION_PROJ][name]).abs().max()
        )
        for name in pristine[TARGET_TEXT_PROJ]
    }
    worst = min(divergence.values())
    if worst < PROJECTOR_DIVERGENCE_MIN:
        raise AssertionError(
            f"text / caption projector の出力差の最小が {worst} — 同じ重みを 2 回読んでいる疑い"
        )
    return {name: round(value, 5) for name, value in divergence.items()}


def _graph_summary(graph: IrGraph, path: Path) -> dict[str, Any]:
    return {
        "nodes": len(graph.nodes),
        "initializers": len(graph.initializers),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "model_bytes": path.stat().st_size,
    }


def export_series(
    model_dir: Path,
    source_dir: Path,
    out_dir: Path,
    *,
    targets: Sequence[str] = TARGETS,
    sym_max: int = SYM_MAX,
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
    cases = build_cases(model_dir, source, text_config, sym_max)
    speaker_max = speaker_sym_max(model_config)
    speaker_cases = build_speaker_cases(int(config.speaker_patched_latent_dim), speaker_max)

    # ---- ① パッチ前の参照（golden の期待値）と方式の実測 ----
    pristine = _pristine_outputs(backbone, projectors, cases)
    static_evidence = _static_scheme_evidence(
        backbone, text_config, model_config, cases, pristine[TARGET_BACKBONE]
    )
    sanity = _sanity(pristine)
    pristine[TARGET_SPEAKER] = _pristine_speaker_outputs(
        speaker_encoder, speaker_norm, speaker_cases
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
        pristine[TARGET_TEXT_PROJ],
        pristine[TARGET_CAPTION_PROJ],
        pristine[TARGET_SPEAKER],
    )
    aux_dim = int(config.duration_aux_dim)
    pristine[TARGET_DURATION] = _pristine_duration_outputs(duration, duration_reference, aux_dim)
    aux_inert = _duration_aux_is_inert(
        duration, duration_reference, aux_dim, pristine[TARGET_DURATION]
    )

    # ---- ② パッチ適用（ここから先で採った eager 値は参照に使えない） ----
    rope_buffers = sorted(
        name for name, _ in backbone.backbone.named_buffers() if name.endswith(ROPE_BUFFER_NAMES)
    )
    patch_irodori.apply_patches()

    # ---- ③ ラッパの eager 同値（パッチ + マスク落としの両方をここで実測する） ----
    graphs = {
        TARGET_BACKBONE: BackboneGraph(backbone),
        TARGET_TEXT_PROJ: ProjectorGraph(projectors[TARGET_TEXT_PROJ]),
        TARGET_CAPTION_PROJ: ProjectorGraph(projectors[TARGET_CAPTION_PROJ]),
        TARGET_SPEAKER: SpeakerGraph(speaker_encoder, speaker_norm, speaker_max),
        TARGET_DURATION: DurationGraph(duration, text_norm),
    }
    graph_args: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        TARGET_BACKBONE: {name: {"input_ids": ids} for name, _kind, ids in cases},
        TARGET_SPEAKER: {name: {"latent": latent} for name, latent in speaker_cases},
        TARGET_DURATION: duration_inputs,
    }
    for target in (TARGET_TEXT_PROJ, TARGET_CAPTION_PROJ):
        graph_args[target] = {
            name: {"hidden": pristine[TARGET_BACKBONE][name]} for name, _kind, _ids in cases
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
    # ターゲット別の `(記号次元名, 上限, 記号を持つ入力の位置)`。テキスト系 3 本と duration は
    # T（512）で、speaker だけ S（参照 latent の patch 後上限）。duration の 2〜5 本目
    # （条件ベクトルと bool）は記号を持たない固定 shape。
    axes: dict[str, tuple[str, int, tuple[int, ...]]] = {
        TARGET_BACKBONE: (TEXT_SYMBOL, sym_max, (0,)),
        TARGET_TEXT_PROJ: (TEXT_SYMBOL, sym_max, (0,)),
        TARGET_CAPTION_PROJ: (TEXT_SYMBOL, sym_max, (0,)),
        TARGET_SPEAKER: (SPEAKER_SYMBOL, speaker_max, (0,)),
        TARGET_DURATION: (TEXT_SYMBOL, sym_max, (0,)),
    }
    written: dict[str, Any] = {}
    for target in targets:
        symbol, upper, dynamic_positions = axes[target]
        target_dir = out_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        by_case = graph_args[target]
        longest = max(by_case, key=lambda name: next(iter(by_case[name].values())).shape[1])
        example = tuple(by_case[longest].values())
        seq = Dim(symbol, min=MIN_SYM_LENGTH, max=upper)
        graph = export_to_file(
            graphs[target],
            example,
            target_dir / MODEL_FILE,
            dynamic_shapes=tuple(
                {1: seq} if index in dynamic_positions else None for index in range(len(example))
            ),
            symbol_names=(symbol,),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        io_files = _write_io(graph, by_case, pristine[target], target_dir)
        written[target] = {
            **_graph_summary(graph, target_dir / MODEL_FILE),
            "io": io_files,
        }
    return {
        "dir": str(out_dir),
        "targets": written,
        "case_lengths": {name: int(ids.shape[1]) for name, _kind, ids in cases},
        "speaker_case_lengths": {name: int(latent.shape[1]) for name, latent in speaker_cases},
        "speaker_sym_max": speaker_max,
        "rope_buffers_lifted": rope_buffers,
        "static_scheme_max_abs": {k: float(f"{v:.3e}") for k, v in static_evidence.items()},
        "no_reference_max_abs": no_reference_max_abs,
        "duration_aux_inert_max_abs": aux_inert,
        "eager_equivalence_max_abs": equivalence,
        "projector_divergence": sanity,
    }


def default_out_root(model_dir: Path) -> Path:
    """生成物の既定の置き場（`outputs/series/irodori-<実重みのディレクトリ名>/`）。

    ターゲット名のサブディレクトリは `export_series` が 1 段掘る（`export_sbv2.py` と同じ形）。
    """
    return SERIES_ROOT / f"irodori-{model_dir.name}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        help="書き出すターゲット（繰り返し指定可。既定は全て）",
    )
    args = parser.parse_args(argv)
    out_dir = default_out_root(args.model_dir) if args.out is None else args.out
    summary = export_series(
        args.model_dir,
        args.source_dir,
        out_dir,
        targets=tuple(args.target) if args.target else TARGETS,
        sym_max=args.sym_max,
    )
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
