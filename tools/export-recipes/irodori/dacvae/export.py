"""実重み DACVAE（Semantic-DACVAE-Japanese-32dim）を IR v1 コンテナ + golden io へ書き出す台本。

recon の G6（decoder）/ G7（encoder）で、Irodori-TTS v4 の**コーデック段**にあたる。テキスト側
6 ターゲットは `irodori/export.py`、ホスト側の latent までの経路は
`irodori/pipeline_ref.py` が持つ。

    cd tools/export-recipes
    uv run --with descript-audiotools --with einops python -m irodori.dacvae.export
    uv run --with descript-audiotools --with einops python -m irodori.dacvae.export --target decoder
    uv run --with descript-audiotools --with einops python -m irodori.dacvae.export --dtype f16
    uv run --with descript-audiotools --with einops python -m irodori.dacvae.export --dtype i8

依存は `--with` で足す（pyproject.toml には入れない — 既定の sync は base deps だけで回る）。
`descript-audiotools` は 2 つの理由で要る: ① `dacvae` パッケージが `audiotools.ml.BaseModel` を
継承する ② encoder の golden 入力が**参照音声を −16 LUFS へ正規化した波形**で、その正規化が
audiotools の実経路（ITU-R BS.1770-4 の IIR）。`einops` は `dacvae/__init__.py` が
`BaseModel.EXTERN` へ名前を足すためだけに import する。

モデル実装は GitHub `facebookresearch/dacvae` の clone（既定 `inputs/irodori/dacvae-src/`）から
**`sys.path` 追加で import** する（`--source-dir`）。固定した版は {@link SOURCE_COMMIT}。

## 何をグラフに載せるか（2 ターゲット・B=1・S / T は記号次元）

| ターゲット | 入力            | 出力          | 中身                                        |
| ---------- | --------------- | ------------- | ------------------------------------------- |
| `decoder`  | `[1,S,32]`      | `[1,1,1920S]` | `quantizer.out_proj` + decoder 主経路       |
| `encoder`  | `[1,T,1920]`    | `[1,T,32]`    | encoder + `quantizer.in_proj` の前半 32 行  |

hop は 1920 サンプル（48kHz / 25 フレーム毎秒）で、チェックポイントの `encoder_rates` の積から
導く（直書きしない）。

### `decoder`（G6）の境界

入力は Irodori の latent と同じ `[1,S,32]`（フレーム軸が記号 S）で、グラフ内で `[1,32,S]` へ
transpose してから `quantizer.out_proj`（conv1d 32→1024 k=1）へ渡す。上流 `DACVAE.decode` の
`out_proj` → `decoder` の 2 段そのもので、`_vae_sample` は通らない（Irodori は
`deterministic_encode=True` で mean をそのまま latent に使う — `codec.py`）。

**透かし枝は上流 README 推奨の②形でバイパスする**（`decoder.alpha = 0.0` に加えて
`decoder.watermark` を `wm_model.encoder_block.forward_no_conv` へ差し替える）。`alpha = 0.0`
だけでは `Decoder.watermark` が入力（96ch）をそのまま返し、**波形にならない** — 波形ヘッド
（`Snake(96)` → `conv1d(96→1,k7)` → `Tanh`）は `wm_model.encoder_block.pre` 側にあるため。
`bypass_watermark` はこの 2 つを両方当て、呼び出し側が出力チャネル数 1 を実測する。

### `encoder`（G7）の境界

`quantizer.in_proj` は conv1d 1024→64（k=1）で、上流は出力を `chunk(2, dim=1)` して前半を
mean・後半を scale に使う。scale は決定的経路では捨てられるので、**前半 32 行 + bias 前半 32 に
切り詰めた conv1d** をグラフに載せる（`_in_proj_truncation_evidence` が `chunk(2)[0]` との
一致を毎回実測する — 許容は `IN_PROJ_TRUNCATION_ATOL`）。

MUST: 入力は**フレーム分割済みの `[1,T,1920]`**（グラフ内で `[1,1,1920T]` へ reshape）。
素直な `[1,1,1920T]` にできないのは、IR の束縛規則が「シンボルは少なくとも 1 つの入力 shape に
**素の形**で現れる」ことを要求する（`karume.verify._check_symbol_bindability`）ため —
`1920T` だけでは T を束縛できず、コンテナを書く前に落ちる。フレーム分割は**要素順を変えない
読み替え**（row-major で フレーム t が サンプル `[1920t, 1920(t+1))` を占める）なので、ホストは
連続バッファをそのまま渡せる。同時に「グラフは 1920 の倍数だけ受ける」契約が shape に出る。

reflect pad（`DACVAE._pad`）と LUFS 正規化・リサンプルは**ホスト責務**でグラフに載せない。
その 2 段の参照値は `irodori/dacvae/host.py` が別に出す。

## 前処理（export の前に必ず通す 2 つ）

- **`nn.utils.remove_weight_norm` を全 `NormConv1d` / `NormConvTranspose1d` へ**。この
  チェックポイントは旧式の `weight_g` / `weight_v` ペアで持っており、畳まないと `weight` が
  実効重みでない（`export_sbv2.py` の `dec` と同じ前処理）。出力のビット一致は
  `_remove_weight_norm_evidence` が毎回実測する。
- **`Snake1d.alpha` をパラメータから素の属性へ降格**（lifted tensor constant）。Snake は
  `x + (α+1e-9).reciprocal() · sin(αx)²` で、`(α+1e-9).reciprocal()` は**定数部分木**だが、
  定数畳み込みの葉として適格なのは lifted 定数だけ（`convert._classify_foldable` —
  パラメータ/バッファ経由は巨大定数の焼き込みを避けるため不適格）。降格しないと
  `reciprocal` が実行時 op として IR に残り、語彙に無いので落ちる。`assert_snake_folded` が
  emit 後の `required_ops` を見て、畳み込みが効いていることを毎回確かめる。

## 常設門（emit の前に全部実測し、1 つでも外れたら**何も書かない**）

1. **主経路の抽出**（`main_path`）が**②形パッチ済み `Decoder`** とビット一致すること。
   グラフに載るのは `Decoder` 本体（写しではない）で、抽出したほうは門でしか使わない —
   「decode が実際に通る層は 28 本で、透かし枝は波形ヘッド 4 本しか寄与しない」という
   構造の主張を、次波（タイル化）が前提にするため。
2. **`in_proj` の切り詰め**が `chunk(2, dim=1)[0]` と一致すること
   （{@link IN_PROJ_TRUNCATION_ATOL} — ビット一致は conv カーネル選択に依存し保証されない）。
3. **`remove_weight_norm` の前後**で decoder / encoder の出力がビット一致すること。
4. **ラッパの eager 同値**（パッチ前の実モジュール出力との差 — {@link EAGER_EQUIV_ATOL} = 0）。
5. **往復の妥当性**（参照音声 → encoder → decoder が有限で、入力波形と相関すること）。
   `_roundtrip_evidence` が相関の下限 {@link ROUNDTRIP_CORRELATION_MIN} を実測で守る。

## 格納 dtype の系列（ADR 0018 / 0019 / 0027 / 0050）

`--dtype f16` / `--dtype i8` はそれぞれ**別系列**（`dacvae-32dim-f16/` / `dacvae-32dim-i8/`）へ
書き出す（同居させると f32 系列の網が圧縮資産へ黙って掛かる）。i8 は per-channel symmetric
（conv1d は軸 0・`ConvTranspose1d` だけ軸 1 — `karume.quantize.QUANT_CHANNEL_AXES` が正本）で、
scale 台帳は {@link _target_scales} がラッパ内 FQN へ張り替えて emit へ渡す。
丸めの順序は `_fake_quant` の MUST — `fold_weight_norm` と
`lift_snake_alphas` の**後**・参照 / golden の採取の**前**で、そのため圧縮系列では参照を
畳み込みの後に**採り直す**（門 3 の「畳む前後のビット一致」は畳む前に採った参照で先に
決着させてから丸める）。

**`Snake1d.alpha` は丸めない** — `lift_snake_alphas` がパラメータから素の属性へ降格した
後に丸めるので `named_parameters` に現れず、emit 側でも重みスロットではない（lifted 定数）
ので **F32 格納のまま残る**。golden も同じ f32 alpha で計算されるので両者は整合する。

golden の入力（decoder の実 latent / encoder の正規化済み波形）は**系列を跨いで同じ**にする
（`--latent-dir` の既定は f32 系列の full-loop golden のまま）— 入力は参照と GPU が同じ
バイト列を読む側で、格納 dtype の影響を受けない。

## 出力レイアウト

    outputs/series/dacvae-32dim/<target>/model.safetensors      重み・定数 + karume_ir
    outputs/series/dacvae-32dim/<target>/io.<case>.safetensors  入力と torch CPU 期待出力

io のテンソルキー規約は他系列と同じ（`input.<グラフ入力名>` / `output.<位置>`）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from karume.artifacts import staged_publication
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import fake_quant_int8, round_weights_to_f16
from karume.shards import resolve_shards

#: 実重みの置き場（`hf download Aratako/Semantic-DACVAE-Japanese-32dim` の展開先を
#: `irodori/dacvae/convert.py` で safetensors 化したもの）。
DEFAULT_MODEL_DIR = INPUTS_ROOT / "irodori" / "dacvae-32dim"

#: モデル実装（GitHub `facebookresearch/dacvae` の clone）の置き場。
DEFAULT_SOURCE_DIR = INPUTS_ROOT / "irodori" / "dacvae-src"

#: 実装 clone の出所と固定した版。`git clone <SOURCE_REPO> inputs/irodori/dacvae-src` で置く。
SOURCE_REPO = "https://github.com/facebookresearch/dacvae"
SOURCE_COMMIT = "414c20785fc3a28373073ea8ef7a1316eeeaca6e"

#: 書き出せる格納 dtype。`i8` は波 2 で足した（conv 支配ネットの i8 品質は前例が無いので、
#: 配布形の席に載せる前に `irodori/measure_quant.py` の `i8-codec-only` が単独で測る —
#: ADR 0050 決定 6 / 量子化 recon の risks）。
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "i8")

WEIGHTS_FILE = "weights.safetensors"
METADATA_FILE = "metadata.json"
MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

TARGET_DECODER = "decoder"
TARGET_ENCODER = "encoder"
TARGETS = (TARGET_DECODER, TARGET_ENCODER)

#: IR に載る記号次元名（グラフごとに閉じているので、両者が同じ名前でも衝突しない）。
DECODER_SYMBOL = "S"
ENCODER_SYMBOL = "T"

#: 記号次元の下限。torch.export の 0/1 特殊化を避けるため 2 で運用する（他系列と同じ）。
MIN_SYM_LENGTH = 2

#: `decoder` の記号次元 S の上限を決める発話長（秒）。生成できる latent の上限そのもの
#: （`irodori.export.DIT_MAX_SECONDS` と同じ 30 秒 — ADR 0047）。**この台本には位置表も
#: マスクも無い**（純粋な畳み込み網）ので、上限を上げてもコンテナは 1 バイトも増えない。
DECODER_MAX_SECONDS = 30.0

#: `encoder` の記号次元 T の上限を決める参照音声長（秒）。チェックポイントの
#: `ref_max_seconds`（120 秒）に合わせる — 参照音声はこれ以上の長さで呼ばれない。
ENCODER_MAX_SECONDS = 120.0

#: 参照音声（公式サンプルの voice cloning 用リファレンス）。48kHz mono・7.6 秒。
DEFAULT_REFERENCE_WAV = INPUTS_ROOT / "irodori" / "v4-small" / "samples" / "clone_ref1.wav"

#: 参照音声の LUFS 正規化の目標（`inference_runtime.SamplingRequest.ref_normalize_db` の既定）。
REFERENCE_NORMALIZE_DB = -16.0

#: 実 latent の供給元（`irodori/pipeline_ref.py` が書く full-loop golden の最終 z）。
DEFAULT_LATENT_DIR = SERIES_ROOT / "irodori-v4-small" / "pipeline"
LATENT_CASE_PREFIX = "case."
LATENT_KEY = "z"

#: ラッパと**パッチ前**の実モジュールの eager 同値に許す差。**0 = ビット一致を要求する**。
#:
#: ラッパがしているのは ① transpose / reshape の付け足し ② `in_proj` の切り詰め
#: ③ `remove_weight_norm` ④ `Snake1d.alpha` の降格の 4 つで、いずれも値を変えない操作
#: （②〜④ は独立した門でもビット一致を実測する）。0 でない値が出たら、どれかの主張が
#: 崩れている — 近似で通さずに落として、どれが崩れたのかを先に確かめる。
EAGER_EQUIV_ATOL = 0.0

#: 往復（encoder → decoder）の再構成波形と入力波形の相関の下限（`_roundtrip_evidence`）。
#:
#: MUST: 恒真にしない — 「有限で非ゼロ」だけでは、チャネル順の取り違えや別経路の混入で
#: 音になっていない出力が通る。実測は 0.988（参照音声 7.6 秒・遅延 0 サンプル）で、
#: 0.8 はその 5 分の 4。学習済みコーデックの再構成なので、経路が正しければ相関は 1 に近い。
ROUNDTRIP_CORRELATION_MIN = 0.8

#: `decoder` の golden ケース `(名前, latent の供給元ケース, 長さ)`。長さ `None` は供給元そのまま。
#:
#: 実 latent は `irodori/pipeline_ref.py` の full-loop golden（`case.full` = 参照音声あり 161
#: フレーム / `case.no-ref` = 参照なし 116 フレーム）から採る。**合成乱数を使わない**のは、
#: Snake の `sin(αx)` が入力の値域に強く効くためで、tolerance の根拠を実運用の値域と
#: 対応させるための選択（`export_sbv2.py` の話者埋め込みと同じ理由）。
#:
#: 長さの両端: 記号次元の下限 2（先頭 2 フレーム）と、**宣言上限そのもの**の 750。
#: 750 は実 latent が足りないのでフレーム方向に反復して作る（値域は実 latent のまま）。
DECODER_CASES: tuple[tuple[str, str, int | None], ...] = (
    ("z-min", "full", MIN_SYM_LENGTH),
    ("z-no-ref", "no-ref", None),
    ("z-full", "full", None),
    ("z-max", "full", 750),
)

#: `encoder` の golden ケース `(名前, フレーム数)`。フレーム数 `None` は参照音声の全長。
#:
#: 入力は参照音声を **−16 LUFS へ正規化 → reflect pad**した波形（ホスト前処理の出口 =
#: グラフの入口）。短尺側は記号次元の下限 2（3840 サンプル）を先頭から切り出す。
ENCODER_CASES: tuple[tuple[str, int | None], ...] = (
    ("wav-min", MIN_SYM_LENGTH),
    ("wav-ref", None),
)


class DacvaeSource:
    """`dacvae` パッケージから取り出す実装（`sys.path` 追加で import する）。

    MUST: import は**実装 clone から**行う（写しを台本に持たない）。`dacvae/__init__.py` は
    `audiotools` と `einops` を引くので、`--with` で足した環境でだけ通る。
    """

    def __init__(self, source_dir: Path) -> None:
        model_py = source_dir / "dacvae" / "model" / "dacvae.py"
        if not model_py.is_file():
            raise SystemExit(
                f"モデル実装が見つからない: {model_py}"
                f"（`git clone {SOURCE_REPO}` の展開先を --source-dir に指定する。"
                f" 固定した版は {SOURCE_COMMIT}）"
            )
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from dacvae.model.dacvae import DACVAE
        from dacvae.nn.layers import Snake1d

        self.dacvae_cls = DACVAE
        self.snake_cls = Snake1d


def read_kwargs(model_dir: Path) -> dict[str, Any]:
    """`irodori/dacvae/convert.py` が書いた `metadata.json` から構成値を読む。

    MUST: HF から config を引き直さない — チェックポイントに埋まっていた `kwargs` が、この
    重みが実際に構成されたときの形の正本。
    """
    path = model_dir / METADATA_FILE
    if not path.is_file():
        raise SystemExit(f"{path} が無い（`uv run python -m irodori.dacvae.convert` で作る）")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "kwargs" not in raw:
        raise SystemExit(f"{path} に 'kwargs' が無い")
    return dict(raw["kwargs"])


def load_codec(source: DacvaeSource, model_dir: Path) -> nn.Module:
    """`metadata.json` の構成で `DACVAE` を組み、safetensors の重みを載せる。

    `load_state_dict(strict=True)` が「317 本を 1 本残らず消費した」ことの門になる
    （`weight_g` / `weight_v` はこの時点ではまだ畳んでいない）。
    """
    weights = model_dir / WEIGHTS_FILE
    if not weights.is_file():
        raise SystemExit(f"{weights} が無い（`uv run python -m irodori.dacvae.convert` で作る）")
    model = source.dacvae_cls(**read_kwargs(model_dir))
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.eval()


def bypass_watermark(decoder: nn.Module) -> None:
    """透かし枝を上流 README 推奨の②形でバイパスする（モジュール docstring の G6 節）。

    MUST: `alpha = 0.0` と `watermark` の差し替えを**両方**当てる。前者だけだと
    `Decoder.watermark` が 96ch の入力をそのまま返し、波形にならない（実測）。後者だけだと
    上流が `alpha` を読む経路が残る。
    """
    decoder.alpha = 0.0
    decoder.watermark = lambda x, message=None, block=decoder.wm_model.encoder_block: (
        block.forward_no_conv(x)
    )


def main_path(decoder: nn.Module) -> nn.Sequential:
    """decode が実際に通る層だけを並べた `Sequential`（**門でのみ使う** — 写し）。

    `DecoderBlock.forward` は `block` を `_chunk_size` ごとに切り、**インデックスが
    `_chunk_size` の倍数のかたまりだけ**を順に通す（残りは透かし枝の up/down サンプラ）。
    末尾は②形バイパスの `forward_no_conv` = `pre[0]`（Snake 96）/ `pre[1]`（conv1d 96→1 k7）/
    `pre[2]`（Tanh）で、`pre[3]` は `Identity` に差し替わる。

    MUST: この写しをグラフに載せない — 載せるのは `Decoder` 本体で、ここは
    `_main_path_evidence` が「主経路はこの 28 本で、透かし枝の残り 25 本は寄与しない」ことを
    実測するためだけに組む（上流が層の並べ方を変えれば、その門が落ちる）。
    """
    layers: list[nn.Module] = [decoder.model[0]]
    for block in decoder.model[1:]:
        size = block._chunk_size
        chunks = [block.block[start : start + size] for start in range(0, len(block.block), size)]
        layers += [
            layer for index, chunk in enumerate(chunks) if index % size == 0 for layer in chunk
        ]
    head = decoder.wm_model.encoder_block.pre
    layers += [head[0], head[1], head[2]]
    return nn.Sequential(*layers)


def fold_weight_norm(model: nn.Module) -> int:
    """全 `weight_norm` を実効重みへ畳む（畳んだ本数を返す）。

    weight_norm が残っている間、`Conv1d.weight` は**実効重みではない**（`weight_g` /
    `weight_v` から forward pre-hook が毎回作る）。そのまま IR へ書き出すと別のモデルになる。
    """
    folded = 0
    for module in model.modules():
        if hasattr(module, "weight_g"):
            nn.utils.remove_weight_norm(module)
            folded += 1
    if folded == 0:
        raise SystemExit(
            "weight_norm を持つモジュールが 1 本も無い — チェックポイントが既に畳まれた形か、"
            "上流が別の正規化に変わっている（実効重みの前提が崩れる）"
        )
    return folded


def lift_snake_alphas(source: DacvaeSource, model: nn.Module) -> int:
    """`Snake1d.alpha` をパラメータから素の属性（lifted tensor constant）へ降格する。

    降格の理由はモジュール docstring の前処理節（`(α+1e-9).reciprocal()` を定数畳み込みの
    frontier に載せるため）。値は変えない（`detach().clone()` するだけ）ので、eager 同値は
    ビット一致のまま通る。

    MUST: 1 本も降格できなければ落とす — 走査が空振りしたまま進むと `reciprocal` が IR に
    残り、語彙に無いので export が落ちる（原因の遠い場所で落ちる形になる）。
    """
    lifted = 0
    for module in model.modules():
        if isinstance(module, source.snake_cls):
            tensor = module._parameters.pop("alpha")
            module.alpha = tensor.detach().clone()
            lifted += 1
    if lifted == 0:
        raise SystemExit("Snake1d が 1 本も見つからない — alpha 降格の走査が空振りしている")
    return lifted


def truncated_in_proj(in_proj: nn.Conv1d) -> nn.Conv1d:
    """`quantizer.in_proj` を**前半 32 行**へ切り詰めた conv1d を作る（mean 側だけ）。

    MUST: `fold_weight_norm` の**後**に呼ぶ（`in_proj.weight` が実効重みである必要がある）。
    `_in_proj_truncation_evidence` が `chunk(2, dim=1)[0]` との一致を実測する
    （許容は `IN_PROJ_TRUNCATION_ATOL`）。
    """
    if hasattr(in_proj, "weight_g"):
        raise SystemExit("in_proj の weight_norm が畳まれていない — 切り詰めが実効重みにならない")
    if in_proj.out_channels % 2 != 0:
        raise SystemExit(
            f"in_proj の出力 {in_proj.out_channels} が偶数でない（mean / scale の 2 分割）"
        )
    half = in_proj.out_channels // 2
    trimmed = nn.Conv1d(
        in_proj.in_channels,
        half,
        kernel_size=in_proj.kernel_size[0],
        stride=in_proj.stride[0],
        padding=in_proj.padding[0],
        dilation=in_proj.dilation[0],
    )
    with torch.no_grad():
        trimmed.weight.copy_(in_proj.weight[:half])
        trimmed.bias.copy_(in_proj.bias[:half])
    return trimmed.eval()


class DecoderGraph(nn.Module):
    """`DACVAE.decode`（`out_proj` → ②形パッチ済み `Decoder`）の export 用ラッパ。

    元の `decode` との差は入口の transpose 1 本だけ（上流は `[B,D,T]` を受け、こちらは
    Irodori の latent と同じ `[B,T,D]` を受ける — `codec.py` の `decode_latent` が
    ホスト側でしている転置をグラフへ入れた形）。**主経路の写しは持たない**（`Decoder` 本体を
    そのまま呼ぶ）。
    """

    def __init__(self, out_proj: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.out_proj = out_proj
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.out_proj(latent.transpose(1, 2)))


class EncoderGraph(nn.Module):
    """`encoder` + 切り詰めた `in_proj` の export 用ラッパ（モジュール docstring の G7 節）。

    元の決定的 encode（`codec.py` の `deterministic_encode` 経路）との差は 3 点:

    - 入力を**フレーム分割済み**`[1,T,1920]` で受け、グラフ内で `[1,1,1920T]` へ戻す
      （要素順を変えない読み替え — 記号 T を素の形で束縛するため）
    - `in_proj(...).chunk(2, dim=1)[0]` を**切り詰めた conv1d** で置き換える
    - 出力を `[1,T,32]` へ転置する（上流も `encode_waveform` の最後で同じ転置をしている）

    reflect pad（`DACVAE._pad`）は**入らない** — ホストが 1920 の倍数に整えてから呼ぶ。
    """

    def __init__(self, encoder: nn.Module, in_proj: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.in_proj = in_proj

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        samples = wav.reshape(1, 1, wav.shape[1] * wav.shape[2])
        return self.in_proj(self.encoder(samples)).transpose(1, 2)


def hop_length(model: nn.Module) -> int:
    """1 latent フレームあたりのサンプル数（`encoder_rates` の積 — 直書きしない）。"""
    hop = int(model.hop_length)
    if hop <= 0:
        raise SystemExit(f"hop_length が {hop} — チェックポイントの encoder_rates が異常")
    return hop


def frame_rate(model: nn.Module) -> float:
    """latent のフレームレート（Hz）。48000 / 1920 = 25。"""
    return int(model.sample_rate) / hop_length(model)


def read_wav(path: Path) -> tuple[torch.Tensor, int]:
    """WAV を mono の `[N]` f32 として読む（`(波形, サンプリング周波数)`）。

    読み手は soundfile。int16 PCM の正規化は **/32768**（int16 の下端 −32768 が厳密に −1.0
    へ写る）で、`irodori/dacvae/host.py` がこの規約を合成 WAV で実測して golden の meta へ記録する。
    """
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).transpose(0, 1)
    if wav.shape[0] != 1:
        # `codec.py` の `encode_waveform` と同じ mono 化（チャネル平均）。
        wav = wav.mean(dim=0, keepdim=True)
    return wav[0].contiguous(), int(sample_rate)


def normalize_reference_wav(wav: torch.Tensor, sample_rate: int, target_db: float) -> torch.Tensor:
    """参照音声を LUFS 正規化する（`codec.py` の `_normalize_loudness` と同じ 2 呼び出し）。

    `AudioSignal.normalize(db)` が ITU-R BS.1770-4 の integrated loudness を測って利得を掛け、
    `ensure_max_of_audio()` が peak > 1 のときだけ縮める。**式を持たずライブラリを呼ぶ**のは、
    K-weighting の IIR 係数とゲーティングを写さないため。上流の staticmethod との
    ビット一致は `irodori/dacvae/host.py` が毎回実測する（そちらは `irodori_tts` を
    import できる）。
    """
    from audiotools import AudioSignal

    signal = AudioSignal(wav.detach().to(torch.float32).unsqueeze(0).unsqueeze(0), int(sample_rate))
    signal.normalize(float(target_db))
    signal.ensure_max_of_audio()
    return signal.audio_data.reshape(-1).contiguous()


def build_decoder_cases(latent_dir: Path, sym_max: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """`decoder` の golden ケース `(名前, latent [1,S,32])`（{@link DECODER_CASES} の表どおり）。

    供給元は `irodori/pipeline_ref.py` の full-loop golden。**足りない長さはフレーム方向の反復**で
    作る（値域を実 latent のまま伸ばすため — 乱数で埋めると Snake の値域が変わる）。
    """
    sources: dict[str, torch.Tensor] = {}
    for name in {source for _n, source, _l in DECODER_CASES}:
        path = latent_dir / f"{LATENT_CASE_PREFIX}{name}{IO_SUFFIX}"
        if not path.is_file():
            raise SystemExit(
                f"実 latent が無い: {path}"
                "（`uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref` で作る）"
            )
        tensors = load_file(str(path))
        if LATENT_KEY not in tensors:
            raise SystemExit(f"{path} に '{LATENT_KEY}' が無い")
        sources[name] = tensors[LATENT_KEY].to(torch.float32)

    cases: list[tuple[str, torch.Tensor]] = []
    for name, source, length in DECODER_CASES:
        latent = sources[source]
        if latent.ndim != 3 or latent.shape[0] != 1:
            raise SystemExit(f"{name}: latent の shape {tuple(latent.shape)} が [1,S,D] でない")
        want = int(latent.shape[1]) if length is None else int(length)
        if not MIN_SYM_LENGTH <= want <= sym_max:
            raise SystemExit(
                f"{name}: S={want} が記号次元の範囲 [{MIN_SYM_LENGTH}, {sym_max}] の外"
            )
        repeats = -(-want // int(latent.shape[1]))
        cases.append((name, latent.repeat(1, repeats, 1)[:, :want].contiguous()))
    return tuple(cases)


def build_encoder_cases(
    model: nn.Module, wav_path: Path, sym_max: int, target_db: float
) -> tuple[tuple[str, torch.Tensor], ...]:
    """`encoder` の golden ケース `(名前, フレーム分割済み波形 [1,T,1920])`。

    参照音声 → LUFS 正規化 → reflect pad（上流 `DACVAE._pad` を**呼ぶ**）→ `[1,T,1920]` の
    読み替え、というホスト前処理の出口そのもの。リサンプルは通らない（48kHz 以外は落とす）。
    """
    wav, sample_rate = read_wav(wav_path)
    if sample_rate != int(model.sample_rate):
        raise SystemExit(
            f"{wav_path.name}: {sample_rate}Hz はコーデックの {int(model.sample_rate)}Hz と違う"
            "（リサンプルはホスト責務で、この台本は通さない）"
        )
    normalized = normalize_reference_wav(wav, sample_rate, target_db)
    with torch.no_grad():
        padded = model._pad(normalized.reshape(1, 1, -1))
    hop = hop_length(model)
    if int(padded.shape[-1]) % hop != 0:
        raise SystemExit(f"reflect pad 後の長さ {int(padded.shape[-1])} が hop {hop} の倍数でない")

    cases: list[tuple[str, torch.Tensor]] = []
    for name, length in ENCODER_CASES:
        frames = int(padded.shape[-1]) // hop if length is None else int(length)
        if not MIN_SYM_LENGTH <= frames <= sym_max:
            raise SystemExit(
                f"{name}: T={frames} が記号次元の範囲 [{MIN_SYM_LENGTH}, {sym_max}] の外"
            )
        if frames * hop > int(padded.shape[-1]):
            raise SystemExit(f"{name}: T={frames} が参照音声の長さを超えている")
        cases.append((name, padded[..., : frames * hop].reshape(1, frames, hop).contiguous()))
    return tuple(cases)


def _decode_reference(model: nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """上流の decode 経路（`out_proj` → ②形パッチ済み `Decoder`）で参照値を採る。"""
    with torch.no_grad():
        return model.decoder(model.quantizer.out_proj(latent.transpose(1, 2)))


def _encode_reference(model: nn.Module, framed: torch.Tensor) -> torch.Tensor:
    """上流の決定的 encode 経路（`encoder` → `in_proj` の `chunk(2)[0]`）で参照値を採る。"""
    with torch.no_grad():
        hidden = model.encoder(framed.reshape(1, 1, -1))
        mean, _scale = model.quantizer.in_proj(hidden).chunk(2, dim=1)
        return mean.transpose(1, 2)


def _pristine_outputs(
    model: nn.Module,
    decoder_cases: Sequence[tuple[str, torch.Tensor]],
    encoder_cases: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """**weight_norm を畳む前**の eager 出力（門 3 の基準であり、f32 系列の golden そのもの）。

    MUST: `fold_weight_norm` の前に呼ぶ。畳んだ後で採ると `_remove_weight_norm_evidence` が
    「畳んだ後どうし」の比較になり、差 0 が自明に成立して偽 PASS する（`export_sbv2.py` の
    `_assert_weight_norm_present` と同じ向きの汚染）。
    """
    if not any(hasattr(module, "weight_g") for module in model.modules()):
        raise AssertionError(
            "weight_norm を畳んだ後に参照を採ろうとした（畳み込みの同値検証が恒真化する）"
        )
    return _eager_outputs(model, decoder_cases, encoder_cases)


def _eager_outputs(
    model: nn.Module,
    decoder_cases: Sequence[tuple[str, torch.Tensor]],
    encoder_cases: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """上流の decode / encode 経路の eager 出力（`weight_norm` の有無を見ない素の計算）。

    圧縮系列では丸めの**後**にもう一度ここを通す（{@link _fake_quant} の順序 MUST）— そちらは
    `weight_g` が既に無いので {@link _pristine_outputs} の門は通せない。門 3（畳む前後の
    ビット一致）は丸める前に決着済みで、この採り直しはその主張を弱めない。
    """
    decoded = {name: _decode_reference(model, latent) for name, latent in decoder_cases}
    for name, audio in decoded.items():
        if audio.shape[1] != 1:
            raise AssertionError(
                f"{name}: decode の出力チャネルが {audio.shape[1]} — 透かしバイパスが②形に"
                "なっておらず波形になっていない（`alpha = 0.0` だけでは 96ch のまま返る）"
            )
    return {
        TARGET_DECODER: decoded,
        TARGET_ENCODER: {name: _encode_reference(model, wav) for name, wav in encoder_cases},
    }


class FakeQuantResult(NamedTuple):
    """{@link _fake_quant} の戻り（丸めの要約と、i8 の per-channel scale 台帳）。"""

    report: str | None
    #: **コーデック本体の FQN** → scale（i8 以外は空）。emit が引くラッパ内 FQN への張り替えは
    #: {@link _target_scales}。
    scales: Mapping[str, torch.Tensor]


#: コーデック本体での `in_proj` の FQN（切り詰め後の scale の出どころ）。
FULL_IN_PROJ_KEY = "quantizer.in_proj.weight"
#: 切り詰めた `in_proj` の**ラッパ内** FQN（`EncoderGraph.in_proj`）。
TRIMMED_IN_PROJ_KEY = "in_proj.weight"

#: ターゲット → scale の出どころ `(本体の FQN 接頭辞, ラッパ内 FQN 接頭辞)` の並び（i8 のみ）。
#:
#: MUST: グラフラッパのコンストラクタと同じ綴りにする — ラッパは本体の部分木を別の属性名で
#: 抱えるので、台帳のキー（本体内 FQN）は emit が引くキー（export したモジュール内 FQN）と
#: 一致しない。食い違えば `_target_scales` か emit のどちらかが fail loudly で落ちる。
TARGET_SCALE_SOURCES: Mapping[str, tuple[tuple[str, str], ...]] = MappingProxyType(
    {
        TARGET_DECODER: (("quantizer.out_proj.", "out_proj."), ("decoder.", "decoder.")),
        # `in_proj` は切り詰めた別モジュールなので接頭辞では張り替えられない
        # （{@link _trimmed_in_proj_scale} が軸 0 の前半だけを切り出す）。
        TARGET_ENCODER: (("encoder.", "encoder."),),
    }
)


def _fake_quant(dtype: str, model: nn.Module) -> FakeQuantResult:
    """格納 dtype の表現可能値へ**実効重み**を丸める（f32 は何もしない）。

    ADR 0006 / 0018 / 0019 / 0027 / 0050 の fake-quant。

    MUST（順序）: ① `fold_weight_norm` の**後**（丸めてから畳むと `weight_g` / `weight_v` を
    丸めることになり、そこから作られる実効重みは f16 / i8 の格子に乗らない）② `lift_snake_alphas`
    の**後**（降格済みの `alpha` は `named_parameters` に現れないので**丸めない** — alpha は
    lifted 定数で emit の重みスロットでもなく F32 格納のまま残るため、golden 側も f32 の
    alpha で計算されているのが正しい対応）③ `truncated_in_proj` の**前**（切り詰めは重みを
    コピーするので、丸める前に作ると encoder だけ元の重みを格納する。i8 では**さらに**、
    scale 台帳が本体側の FQN でしか採れないため、丸めが先でないと台帳（本体の前半行 —
    {@link _trimmed_in_proj_scale}）と切り詰めコピーの格納重みが別の重みを指す）
    ④ 参照・golden の採取の**前**（`karume.quantize` の MUST）。
    """
    if dtype == "f32":
        return FakeQuantResult(None, {})
    if dtype == "i8":
        int8 = fake_quant_int8(model)
        print(f"[fake-quant] codec: i8 per-channel へ丸めた — {int8.describe()}", flush=True)
        return FakeQuantResult(int8.describe(), int8.scales)
    report = round_weights_to_f16(model)
    print(f"[fake-quant] codec: f16 表現可能値へ丸めた — {report.describe()}", flush=True)
    return FakeQuantResult(report.describe(), {})


def _trimmed_in_proj_scale(scales: Mapping[str, torch.Tensor], in_proj: nn.Module) -> torch.Tensor:
    """切り詰めた `in_proj` の per-channel scale（本体の scale の**前半行**そのもの）。

    `truncated_in_proj` は丸めた後の重みの前半 32 行をコピーするだけなので、per-channel 軸
    （`Conv1d` は出力チャネル = 軸 0）の scale も前半を切れば厳密に対応する。

    MUST: 切り詰めた重みから scale を**引き直さない** — f32 の割り算で 1ulp 動きうるので、
    そのときは emit の逆変換ビット一致検査が落ちる（ADR 0019 の「scale は fake-quant が
    使った値そのまま」）。
    """
    source = scales.get(FULL_IN_PROJ_KEY)
    if source is None:
        raise SystemExit(f"scale 台帳に '{FULL_IN_PROJ_KEY}' が無い（切り詰めの対応が取れない）")
    rows = int(in_proj.out_channels)
    if int(source.shape[0]) < rows:
        raise SystemExit(
            f"'{FULL_IN_PROJ_KEY}' の scale 行数 {int(source.shape[0])} が切り詰め後の"
            f" {rows} に足りない（per-channel 軸の取り違え）"
        )
    return source[:rows]


def _target_scales(
    target: str, wrapper: nn.Module, scales: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """scale 台帳をそのターゲットの**ラッパ内 FQN**へ張り替える（{@link TARGET_SCALE_SOURCES}）。

    落とすのは**ラッパに無い重み**だけ（透かし枝の残りや反対側のターゲット）。逆向き
    （ラッパにあるのに台帳に無い）は emit が fail loudly で落とす。
    """
    if not scales:
        return {}
    owned = {name for name, _ in wrapper.named_parameters()}
    picked: dict[str, torch.Tensor] = {}
    for source, destination in TARGET_SCALE_SOURCES[target]:
        for key, scale in scales.items():
            if not key.startswith(source):
                continue
            rebased = destination + key[len(source) :]
            if rebased in owned:
                picked[rebased] = scale
    if target == TARGET_ENCODER:
        picked[TRIMMED_IN_PROJ_KEY] = _trimmed_in_proj_scale(scales, wrapper.in_proj)
    if not picked:
        raise SystemExit(
            f"{target}: scale 台帳が 1 本もラッパ内 FQN へ張り替えられなかった"
            f"（{TARGET_SCALE_SOURCES[target]} の接頭辞がラッパの構成と食い違っている）"
        )
    return picked


def _main_path_evidence(
    decoder: nn.Module, cases: Sequence[tuple[str, torch.Tensor]], out_proj: nn.Module
) -> dict[str, float]:
    """主経路の抽出が②形パッチ済み `Decoder` と**ビット一致**することを実測する（門 1）。

    MUST: 恒真にしない — 抽出（`main_path`）と `Decoder.forward` は別の経路で、前者は
    `_chunk_size` の倍数だけを拾う写し、後者は上流本体。一致は「透かし枝の残り 25 本が
    decode に寄与しない」ことの実測であって、自明ではない。
    """
    extracted = main_path(decoder)
    evidence: dict[str, float] = {}
    for name, latent in cases:
        with torch.no_grad():
            embedded = out_proj(latent.transpose(1, 2))
            reference = decoder(embedded)
            actual = extracted(embedded)
        if not torch.equal(reference, actual):
            raise AssertionError(
                f"{name}: 主経路の抽出が Decoder 本体と一致しない"
                f"（最大絶対差 {float((reference - actual).abs().max())}）"
                " — 透かし枝の切り分け（`_chunk_size` の倍数だけを通す）が崩れている"
            )
        evidence[name] = 0.0
    return evidence


#: 門 2（in_proj 切り詰め）の許容絶対差。
#:
#: | 系   | 素実測（max abs）       | 採用値（≈10 倍） |
#: | ---- | ----------------------- | ---------------- |
#: | 門 2 | 2.9802e-8（2026-08-13） | 3e-7             |
#:
#: MUST: ビット一致（`torch.equal`）を要求しない — 比べる 2 辺は「out_channels 2h の conv の
#: 前半 h」と「out_channels h の conv」で、torch の conv はチャネル数・スレッド数でカーネルの
#: ブロッキング（= 縮約順）を変えるため、ビット一致は仕様保証の無い実装挙動（実測でも
#: スレッド数 6 で 1 ulp 差・8 で 0 と揺れる — テスト追加で global RNG 列がずれた途端に
#: 割れた）。門の目的は mean / scale の取り違え検出で、取り違えの差は O(1) — この許容で
#: 判別力は落ちない（取り違えが落ち続けることは `tests/test_irodori/dacvae/export.py` が固定する）。
IN_PROJ_TRUNCATION_ATOL = 3e-7


def _in_proj_truncation_evidence(
    model: nn.Module, trimmed: nn.Conv1d, cases: Sequence[tuple[str, torch.Tensor]]
) -> dict[str, float]:
    """切り詰めた `in_proj` が `chunk(2, dim=1)[0]` と一致することを実測する（門 2）。

    MUST: 恒真にしない — 切り詰めは「出力チャネルの前半 = mean」という上流の分割規約に
    乗っており、後半（scale）を取る取り違えは shape も dtype も一致するのでここでしか出ない。
    戻り値には実測した最大絶対差をそのまま載せる（許容は {@link IN_PROJ_TRUNCATION_ATOL}）。
    """
    evidence: dict[str, float] = {}
    for name, wav in cases:
        with torch.no_grad():
            hidden = model.encoder(wav.reshape(1, 1, -1))
            expected, _scale = model.quantizer.in_proj(hidden).chunk(2, dim=1)
            actual = trimmed(hidden)
        diff = float((expected - actual).abs().max())
        if diff > IN_PROJ_TRUNCATION_ATOL:
            raise AssertionError(
                f"{name}: 切り詰めた in_proj が chunk(2)[0] と一致しない"
                f"（最大絶対差 {diff} > 許容 {IN_PROJ_TRUNCATION_ATOL}）"
            )
        evidence[name] = diff
    return evidence


def _remove_weight_norm_evidence(
    model: nn.Module,
    decoder_cases: Sequence[tuple[str, torch.Tensor]],
    encoder_cases: Sequence[tuple[str, torch.Tensor]],
    pristine: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    """`remove_weight_norm` の前後で出力がビット一致することを実測する（門 3）。

    畳み込みは `w = g · v/‖v‖` の合成で、数学的には恒等でも**浮動小数では別の丸め**になりうる。
    ここが 0 でなければ golden の期待値（畳む前に採った値）は畳んだ後のグラフの期待値として
    使えない — 緩めずに落とす。
    """
    if any(hasattr(module, "weight_g") for module in model.modules()):
        raise AssertionError("weight_norm がまだ残っている（畳んだ後で呼ぶ門）")
    evidence: dict[str, dict[str, float]] = {}
    for target, cases, call in (
        (TARGET_DECODER, decoder_cases, _decode_reference),
        (TARGET_ENCODER, encoder_cases, _encode_reference),
    ):
        measured: dict[str, float] = {}
        for name, value in cases:
            actual = call(model, value)
            expected = pristine[target][name]
            if not torch.equal(expected, actual):
                raise AssertionError(
                    f"{target}/{name}: remove_weight_norm の前後で出力が変わった"
                    f"（最大絶対差 {float((expected - actual).abs().max())}）"
                )
            measured[name] = 0.0
        evidence[target] = measured
    return evidence


def _roundtrip_evidence(
    encoder_graph: nn.Module, decoder_graph: nn.Module, framed: torch.Tensor
) -> dict[str, float]:
    """参照音声の往復（encode → decode）が波形として妥当であることを実測する（門 5）。

    見るのは ① 出力が全て有限 ② 入力波形との相関が {@link ROUNDTRIP_CORRELATION_MIN} 以上、
    の 2 点。2 本のグラフを**繋いで**通す唯一の門で、片方だけの取り違え（チャネル順・
    転置の向き・in_proj の前半後半）はここで相関として落ちる。
    """
    with torch.no_grad():
        latent = encoder_graph(framed)
        audio = decoder_graph(latent)
    if not bool(torch.isfinite(audio).all()):
        raise AssertionError("往復の出力に非有限値がある")
    source = framed.reshape(-1)
    restored = audio.reshape(-1)
    if restored.numel() != source.numel():
        raise AssertionError(
            f"往復の長さが {restored.numel()} で入力 {source.numel()} と違う（hop の取り違え）"
        )
    centered_source = source - source.mean()
    centered_restored = restored - restored.mean()
    denominator = float(centered_source.norm() * centered_restored.norm())
    if denominator == 0.0:
        raise AssertionError("往復の出力が定数（相関を測れない）")
    correlation = float((centered_source * centered_restored).sum()) / denominator
    if correlation < ROUNDTRIP_CORRELATION_MIN:
        raise AssertionError(
            f"往復の相関 {correlation:.4f} が下限 {ROUNDTRIP_CORRELATION_MIN} を割った"
            " — 経路のどこかで別の値を通している"
        )
    return {
        "correlation": round(correlation, 5),
        "input_rms": round(float(source.pow(2).mean().sqrt()), 6),
        "output_rms": round(float(restored.pow(2).mean().sqrt()), 6),
        "latent_abs_max": round(float(latent.abs().max()), 5),
    }


def _check_wrapper_equivalence(
    wrapper: nn.Module, argument: torch.Tensor, expected: torch.Tensor, where: str, atol: float
) -> float:
    """ラッパの出力が参照の実モジュール出力と一致することを見る（門 4）。

    参照は f32 系列なら**畳む前**、圧縮系列なら**丸めた後に採り直したもの**（どちらの場合も
    ラッパと同じ重みで計算された値なので、要求は `EAGER_EQUIV_ATOL = 0` のまま）。
    """
    with torch.no_grad():
        actual = wrapper(argument)
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"{where}: ラッパの出力 shape {tuple(actual.shape)} が"
            f"期待 {tuple(expected.shape)} と違う"
        )
    diff = float((actual - expected).abs().max())
    if diff > atol:
        raise AssertionError(f"{where}: eager 同値が崩れた（最大絶対差 {diff} > {atol}）")
    return diff


def assert_snake_folded(graph: IrGraph, where: str) -> None:
    """Snake の定数部分木が畳まれていることを emit 後の `required_ops` で確かめる。

    `reciprocal` / `pow` が残っていたら `alpha` の降格が効いていない（IR の語彙にも無いので
    本来は変換段で落ちるが、語彙が増えたときに黙って実行時 op へ落ちる形になりうる）。
    """
    leaked = sorted({"reciprocal", "pow", "div"} & set(graph.required_ops))
    if leaked:
        raise AssertionError(
            f"{where}: Snake の定数部分木が畳まれず {leaked} が実行時 op として残っている"
            " — `Snake1d.alpha` の降格（lifted tensor constant）が効いていない"
        )
    if "sin" not in graph.required_ops:
        raise AssertionError(f"{where}: sin がグラフに無い — Snake が丸ごと消えている疑い")


def _write_io(
    graph: IrGraph,
    inputs: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    out_dir: Path,
) -> list[str]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    どちらのターゲットも 1 入力 1 出力なので、名前と本数の突合はグラフ宣言と直接行う。
    """
    declared = [spec.name for spec in graph.inputs]
    if len(declared) != 1 or len(graph.outputs) != 1:
        raise AssertionError(
            f"入力 {declared} / 出力 {list(graph.outputs)} が 1 本ずつでない（io の規約が違う）"
        )
    written: list[str] = []
    for name, value in inputs.items():
        tensors = {
            f"{INPUT_PREFIX}{declared[0]}": normalize_boundary_tensor(
                value.detach().contiguous(), f"{name} の入力 '{declared[0]}'"
            ),
            f"{OUTPUT_PREFIX}0": normalize_boundary_tensor(
                expected[name].detach().contiguous(), f"{name} の出力 0"
            ),
        }
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


class TargetAxis(NamedTuple):
    """ターゲット別の記号次元の宣言（`dynamic` は記号を載せる入力の軸番号）。"""

    symbol: str
    upper: int
    axis: int


def _graph_summary(graph: IrGraph, path: Path) -> dict[str, Any]:
    return {
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "inputs": [[spec.name, list(spec.shape)] for spec in graph.inputs],
        "output_shapes": [list(graph.values[name].shape) for name in graph.outputs],
        "model_bytes": sum(p.stat().st_size for p in resolve_shards(path)),
    }


def export_series(
    model_dir: Path,
    source_dir: Path,
    out_dir: Path,
    *,
    targets: Sequence[str] = TARGETS,
    latent_dir: Path = DEFAULT_LATENT_DIR,
    reference_wav: Path = DEFAULT_REFERENCE_WAV,
    normalize_db: float = REFERENCE_NORMALIZE_DB,
    dtype: str = "f32",
) -> dict[str, Any]:
    """IR コンテナと golden io を書き、要約を返す。

    MUST: 生成物は**ターゲットごとの作業席**へ書き、全ての門（snake 畳み込み・境界正規化）を
    通してから据える。門より前に final へ置くと、落ちた実走が「検収門を通れる資産」を残す —
    io golden は同じ壊れたグラフから採るので互いに整合し、TS 側の突合は**緑になる**
    （「いつ公開してよいか」の綴りは {@link _shared.decode_series._publish}・据え替えと
    後片付けの規律は core の原語 {@link karume.artifacts.staged_publication}）。
    """
    source = DacvaeSource(source_dir)
    model = load_codec(source, model_dir)
    bypass_watermark(model.decoder)
    rate = frame_rate(model)
    decoder_max = int(DECODER_MAX_SECONDS * rate)
    encoder_max = int(ENCODER_MAX_SECONDS * rate)

    decoder_cases = build_decoder_cases(latent_dir, decoder_max)
    encoder_cases = build_encoder_cases(model, reference_wav, encoder_max, normalize_db)

    # ---- ① weight_norm を畳む前の参照（golden の期待値）と主経路の実測 ----
    pristine = _pristine_outputs(model, decoder_cases, encoder_cases)
    main_path_evidence = _main_path_evidence(model.decoder, decoder_cases, model.quantizer.out_proj)

    # ---- ② 前処理（畳み込み → alpha 降格）と、それが値を変えていないことの実測 ----
    folded = fold_weight_norm(model)
    fold_evidence = _remove_weight_norm_evidence(model, decoder_cases, encoder_cases, pristine)
    lifted = lift_snake_alphas(source, model)
    # MUST: 丸めは畳み込みと alpha 降格の後・切り詰めと参照採り直しの前（`_fake_quant` の順序）。
    quantized = _fake_quant(dtype, model)
    if quantized.report is not None:
        # 圧縮系列の golden は**丸めた重み**で採り直す（丸め前の pristine は門 3 で使い切った）。
        pristine = _eager_outputs(model, decoder_cases, encoder_cases)
    trimmed = truncated_in_proj(model.quantizer.in_proj)
    truncation_evidence = _in_proj_truncation_evidence(model, trimmed, encoder_cases)

    # ---- ③ ラッパの eager 同値（前処理 3 つをまとめてここで実測する） ----
    graphs = {
        TARGET_DECODER: DecoderGraph(model.quantizer.out_proj, model.decoder),
        TARGET_ENCODER: EncoderGraph(model.encoder, trimmed),
    }
    graph_args: dict[str, dict[str, torch.Tensor]] = {
        TARGET_DECODER: dict(decoder_cases),
        TARGET_ENCODER: dict(encoder_cases),
    }
    equivalence = {
        target: max(
            _check_wrapper_equivalence(
                module, value, pristine[target][name], f"{target}/{name}", EAGER_EQUIV_ATOL
            )
            for name, value in graph_args[target].items()
        )
        for target, module in graphs.items()
    }
    roundtrip = _roundtrip_evidence(
        graphs[TARGET_ENCODER], graphs[TARGET_DECODER], dict(encoder_cases)[ENCODER_CASES[-1][0]]
    )

    # ---- ④ export と golden の書き出し ----
    axes = {
        TARGET_DECODER: TargetAxis(DECODER_SYMBOL, decoder_max, 1),
        TARGET_ENCODER: TargetAxis(ENCODER_SYMBOL, encoder_max, 1),
    }
    written: dict[str, Any] = {}
    for target in targets:
        axis = axes[target]
        target_dir = out_dir / target
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        by_case = graph_args[target]
        longest = max(by_case, key=lambda name: by_case[name].shape[axis.axis])
        sequence = Dim(axis.symbol, min=MIN_SYM_LENGTH, max=axis.upper)
        with staged_publication(target_dir) as staged:
            # ディレクトリの席は書き手が作る（原語は席を作らない — path しか渡さない）。
            staged.mkdir()
            graph = export_to_file(
                graphs[target],
                (by_case[longest],),
                staged / MODEL_FILE,
                dynamic_shapes=({axis.axis: sequence},),
                symbol_names=(axis.symbol,),
                weight_dtype=dtype,
                weight_scales=_target_scales(target, graphs[target], quantized.scales),
            )
            assert_snake_folded(graph, target)
            io_files = _write_io(graph, by_case, pristine[target], staged)
        written[target] = {**_graph_summary(graph, target_dir / MODEL_FILE), "io": io_files}
    return {
        "dir": str(out_dir),
        "dtype": dtype,
        "fake_quant": quantized.report,
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "targets": written,
        "hop_length": hop_length(model),
        "frame_rate": rate,
        "decoder_sym_max": decoder_max,
        "encoder_sym_max": encoder_max,
        "decoder_case_lengths": {name: int(z.shape[1]) for name, z in decoder_cases},
        "encoder_case_frames": {name: int(w.shape[1]) for name, w in encoder_cases},
        "reference_wav": str(reference_wav),
        "normalize_db": normalize_db,
        "weight_norm_folded": folded,
        "snake_alphas_lifted": lifted,
        "main_path_max_abs": main_path_evidence,
        "weight_norm_fold_max_abs": fold_evidence,
        "in_proj_truncation_max_abs": truncation_evidence,
        "eager_equivalence_max_abs": equivalence,
        "roundtrip": roundtrip,
    }


def default_out_root(model_dir: Path, dtype: str = "f32") -> Path:
    """生成物の既定の置き場（`outputs/series/<実重みのディレクトリ名>{,-f16,-i8}/`）。

    ターゲット名のサブディレクトリは `export_series` が 1 段掘る（他系列と同じ形）。

    MUST: dtype ごとに別ディレクトリ（ADR 0018 / 0019 / 0027）— 綴りは `karume.dist` の
    `IRODORI_CODEC_NAME` + dtype 接尾と一致させる（書き手と読み手が同じ 1 語から組む）。
    """
    suffix = "" if dtype == "f32" else f"-{dtype}"
    return SERIES_ROOT / f"{model_dir.name}{suffix}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR)
    parser.add_argument("--reference-wav", type=Path, default=DEFAULT_REFERENCE_WAV)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は --dtype ごとの系列 —"
        " outputs/series/<--model-dir のディレクトリ名>{,-f16,-i8}/）",
    )
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
        latent_dir=args.latent_dir,
        reference_wav=args.reference_wav,
        dtype=args.dtype,
    )
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
