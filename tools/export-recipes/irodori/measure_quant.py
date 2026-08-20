"""Irodori-TTS v4 の量子化構成別 品質劣化を torch CPU で実測する（波 2 の測定台本）。

`measure_quant_sbv2.py` / `measure_quant_anima.py` と同じ思想の品質ゲート — GPU コードは 0 行で、
ここで測るのは**量子化そのものの質**（ADR 0006 の fake-quant 方法論では E2E は実装誤差しか
測らないため、品質は別軸で測る必要がある）。答えを出したい問いは 1 つ:

    **S（フレーム数）は動くか** — duration を i8 にすると発話長が動きうる（SBV2 w8 の
    w_ceil 198→196 と同じ軸・ADR 0029）。latent 門の「S / forwards 完全一致」が壊れるなら、
    配布形の w8 席は混成（duration だけ据え置き）にする必要がある（ADR 0050 決定 6）。

    cd tools/export-recipes
    uv run --with descript-audiotools --with einops --with 'transformers==5.14.1' \
        python -m irodori.measure_quant --config f32
    uv run ... python -m irodori.measure_quant --config i8-all
    uv run ... python -m irodori.measure_quant --config i4-all
    uv run ... python -m irodori.measure_quant --config gptq-rtn   # 校正付き（波 J-2）
    uv run ... python -m irodori.measure_quant --scan   # w4 の対象規模と試算だけ（実測なし）
    uv run ... python -m irodori.measure_quant          # 実測はせずレポートだけ組み直す

**1 実行 = 1 構成**（`--config`）。full-loop は DiT を 1 ケースあたり 240 forward 回す torch CPU
実行なので、9 構成を 1 プロセスに詰めると数時間の不可分な塊になる。構成ごとに独立に呼べる形に
して、`--out` の下に per-config の JSON / WAV / z を残し、最後の実行が全部を突き合わせて
`report.json` と Markdown 表を出す（基準 `f32` を先に回すこと — 他構成はその成果物を読む）。

## 構成（`--config` の綴り）

| 構成             | 重み                 | 何を答える                                      |
| ---------------- | -------------------- | ----------------------------------------------- |
| `f32`            | 丸めなし（**基準**） | 恒真化防止（full-loop golden の z とバイト一致）|
| `f16`            | 全役割 f16           | ADR 0050 波 1 の系列の対応物                    |
| `i8-all`         | 全役割 i8            | 配布 `w8` 席（配布 25.2%）                      |
| `i8-mixed`       | duration 以外 i8     | 混成表の候補（ADR 0050 決定 6 の (i) 案）       |
| `w8a8`           | 全役割 i8 + 活性 i8  | 配布 `w8a8` 席（DiT の Linear だけ活性も i8）   |
| `i8-<役割>-only` | その役割だけ i8      | **直交分解 5 本**（境界 = `load_*` と 1:1）     |

w4（ADR 0069 追記 5）の 5 構成は**スクリーニングの勝者 4 方式**（g=32 固定）を全役割へ当てた
もので、丸めの対象 op を i8 と同じ 5 種へ広げる（`op_types` の明示 opt-in — linear / conv 系 /
embedding）。**出荷できる形ではない**（i4 の実行経路は linear 限定 = 決定 5・conv 系の scale は
emit へ構造的に渡せない）— ここで測るのは「非 linear まで丸めたときに品質がどこまで戻るか」
の上限で、格納の受理集合も runtime も 1 行も触らない。

| 構成                | 方式          | 何を答える                                        |
| ------------------- | ------------- | ------------------------------------------------- |
| `i4-all`            | `rtn-i4-g32`  | 全役割 w4 で **S が動くか**（第一の門）+ 品質      |
| `i4-mixed`          | `rtn-i4-g32`  | `i8-mixed` と同じ軸の受け皿（duration だけ据え置き）|
| `nf4-all`           | `nf4`         | 正規分位点の格子でどこまで戻るか                  |
| `mxfp4-all`         | `mxfp4`       | 2 のべき scale（E8M0）の取り分 — 4.25 bpw         |
| `kmeans-shared-all` | `kmeans:shared` | 学習した共有表の取り分（表 16×f32 が別途要る）  |

`i4-mixed` を RTN にだけ置くのは、S が動くかどうかが**丸めの粗さ**で決まるから（4 方式ぶんの
混成を並べても同じ問いを 4 回聞くだけになる。RTN で受け皿の形が立てば他方式へ横展開できる）。

`kmeans-shared-all` は表を張る前に対象の正規化値を 1 本へ連結するので、dit 役割（丸める木で
7.5 億要素）の全量 fit は f64 の作業領域まで含めて 20GB 超になる。`--kmeans-fit-stride` が
core の逃げ道（**表の fit だけ**等間隔部分標本・適用は常に全量）で、使った値は出力へ載る。

## 校正付き丸め（{@link CALIB_CONFIGS}・波 J-2）

上の w4 5 構成は重みだけを見て丸める（RTN 系）。**校正付き丸め**（GPTQ — core の
`karume.quant_calib`）は「その層に実際に流れる活性」から**同じ格子の中で**丸め先を選び直す
側で、格納グリッド 3 種を同じ測定列へ足す:

| 構成          | 方式 / 格子 | 何を答える                                        |
| ------------- | ----------- | ------------------------------------------------- |
| `gptq-rtn`    | `gptq/rtn`  | 出荷可能な格子のまま校正でどこまで戻るか（本命）  |
| `gptq-nf4`    | `gptq/nf4`  | 正規分位点の格子 × 校正                           |
| `gptq-kmeans` | `gptq/kmeans_shared` | 学習した表（**射程は層内**）× 校正        |

対象は **DiT block 列の `nn.Linear` 限定**（{@link CALIB_TARGET}）で、`i4-linear`（全役割の
linear）とは**集合が違う** — 校正の駆動（`calibrate_stages`）は stage 内の `nn.Linear` に
閉じており、backbone / projector / speaker / duration / codec も、DiT の `in_proj` /
`out_proj` / `cond_module` も stage の外にあるため。

校正入力は **{@link irodori.pipeline_ref.PIPELINE_CASES} 2 件の参照ループ（f32・CFG 込み）の
全 step**で、step ごとに先頭 DiT block への `(hidden, 付随引数)` を捕まえて 1 バッチにする
（{@link capture_case_batches}）。拡散モデルの活性は t で分布が動くので、**step を横断して**
採らないと後半 step の分布だけ校正から漏れる。付随引数（`cond_embed` / 条件 state /
マスク / RoPE 表）は `DitGraph.forward` がループの**前**に 1 回作って全 block へ同じものを
渡すので、stage 間で不変にできる（`advance_kwargs` は要らない）。

MUST: 捕捉は**丸めを 1 本も当てる前**（pristine）に 1 回だけ。stage 分解が `DitGraph` の
1 forward と**ビット一致**することも、丸める前にその場で実測する
（{@link assert_stage_split_matches_graph}）— block 本体の写しが上流とずれると、値が静かに
別物になったまま表だけ出る。

**linear 限定形は構成として置かない** — `--scan` が対象規模（本数・チャネル・要素・縮小率）を
数えて出す。conv 主体なら linear だけ丸めても配布サイズがほとんど動かず、full-loop を数十分
回す価値が無いため、回すかどうかをこの表で決める。

量子化軸（`channel_rows` で畳んだ最終次元）が g32 で割り切れない重みは `include` で対象から
外す（端数 group を作らない = ADR 0069 決定 2）。**除外一覧は毎回出力へ載せる** — 黙って外すと
「全役割を丸めた」と読める表の裏で、割り切れない層だけ f32 のまま残る。

サイズ列（実効 bpw / 投影 MiB / 式）は**式による投影**で実測ではない。計数は**配布グラフに
載る重みだけ**（`irodori.export.load_dit` はグラフに載らない backbone のコピーまで抱えるので、
丸めた木で数えると出荷しない重みのぶんだけ膨らむ）。丸める木のほうは i8 と同じ役割モジュール
に据え置く — 「何を丸めたか」が i8 と食い違うと方式間の比較が成立しない。

`w8a8` は `i8-all` と**同じ丸めの重み**に、DiT の `nn.Linear` 入力へ per-token i8 の
fake-quant（`karume.act_quant.quantize_rows` — ランタイム `quantize-rows.ts` の数値鏡像）を
モジュールフックで足したもの。フックで安全なのは `patch.py` が `functional` で呼ぶのが
`rms_norm` / `silu` だけで、量子化対象の Linear が実モジュールのまま残るため（SBV2 では
`functional.conv1d` 直呼びのせいでフックが沈黙で取りこぼした — ADR 0029 の検出限界）。
**適用本数は毎回出力し、0 本なら fail loudly**（「w8a8 のつもりで w8 の数を採った」に気づけない
形を作らない）。活性量子化を DiT に限るのは配布形の席と揃えるため（`session` は `dit` の
Session にだけ降りる — models 側 `pipeline.ts`）。

直交分解の 5 本は `dit` / `backbone` / `speaker` / `duration` / `codec`。**5 本の和は `i8-all`
にならない** — projector 2 本（各 6.83MB）は単独の軸に立てていない（桁が 2 つ小さく、単独で
量子化する動機が無い — 量子化 recon の sizeBreakdown）。`i8-all` / `i8-mixed` には入る。

## 時間グリッドの固定（SNR / LSD を意味のある量にするための MUST）

duration の出力は S を決めるので、量子化で銀行家丸めが 1 つ飛ぶだけで **latent 長そのものが
変わり**、波形長が構成間で食い違って SNR が定義できなくなる。そこで基準以外の構成は
**`f32` の S を外から固定して**回す（`irodori.pipeline_ref.run_case(frames_override=…)` —
`measure_quant_sbv2.py` が dump 側 `w_ceil` で時間グリッドを固定するのと同じ選択）。

**構成が自分で予測した S** は `predictedS` として別に残す。これが**この台本の主成果**で、
品質指標は「同じ時間グリッド上での比較」として読む — 実運用では発話長も動く。

## 直交性の門（恒真化の遮断）

役割の効き方は構造的に決まっているので、**「変わること」だけでなく「変わらないこと」も**
門にする（片側だけだと、丸めが素通りしても scope が漏れても緑になる）:

- `duration` は **S にしか効かない**（`log_frames` → 銀行家丸め → S。他の段は読まない）
  ⇒ グリッドを固定した `i8-duration-only` の z と波形は基準と**ビット一致**するのが正しい
- `codec` は **波形にしか効かない**（latent の後段）⇒ z はビット一致・波形は変わる
- `speaker` は**参照のあるケースだけ** z に効く（参照なしはホストのゼロ供給短絡で encoder が
  走らない — ADR 0048）⇒ `i8-speaker-only` は full で変わり・no-ref でビット一致、が緑の条件
- それ以外（backbone / projector / dit）は z が変わる
- `w8a8` は**重みだけの `i8-all` とも z が違う**（活性シムの素通り検出 — 素通りしたら
  z は `i8-all` とビット一致してしまい、基準との差だけ見ていると「効いている」と誤読する）

## 出力

    <--out>/<config>.json               構成 1 本ぶんの実測（S / forwards / 秒 / 品質指標）
    <--out>/<config>.<case>.safetensors z と波形（次の構成が基準として読む f32 の正本）
    <--out>/<config>.<case>.wav         聴き比べ用（16bit PCM・最終裁定は聴感 = ユーザー）
    <--out>/report.json                 全構成の突き合わせ（Markdown 表は stdout）

NOTE: 末尾トリム（`find_flattening_point`）と秒指定の切り出しは**通さない** — 構成ごとに
切る位置が動くと波形長が揃わず SNR が定義できない。実運用の長さは `predictedSeconds` が持つ。

NOTE: 指標の実装（`snr_db` / `log_spectral_distance` / `rel_rms`）は `measure_quant_sbv2.py`
からの**複製**。共有モジュール化は exporter の io 共通化波（Gemma 直前）の領分で、そこまでは
台本ごとに閉じた形を保つ（2026-08-12 網羅レビューの D3 裁定）。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from _shared.paths import OUTPUTS_ROOT, SERIES_ROOT
from karume import act_quant
from karume.quant_calib import (
    CalibMethod,
    CalibReport,
    GridSpec,
    StageBatch,
    StageSpec,
    calibrate_stages,
)
from karume.quant_methods import (
    DEFAULT_CODEBOOK_LEVELS,
    fake_quant_kmeans,
    fake_quant_mxfp4,
    fake_quant_nf4,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QUANT_MODULE_TYPES,
    fake_quant_int4,
    iter_quant_targets,
)

from . import export as ex
from . import patch
from . import pipeline_ref as ip
from .dacvae import export as dv

#: デモ・ベンチの生成物置き場（資産と分離する — `rm -rf outputs/demo` が系列を巻き込まない）。
DEFAULT_OUT = OUTPUTS_ROOT / "demo" / "irodori-quant-sim"

#: 恒真化防止の突合先（`irodori/pipeline_ref.py` が f32 系列へ書く full-loop golden）。
DEFAULT_GOLDEN_DIR = SERIES_ROOT / "irodori-v4-small" / "pipeline"

#: 基準構成の綴り（他構成はこの成果物を読んで比較する）。
BASE_CONFIG = "f32"

#: 役割 = グラフ境界（`irodori.export.load_*` と 1:1）+ コーデック。
ROLE_CODEC = "codec"
ROLES: tuple[str, ...] = (
    ex.TARGET_BACKBONE,
    ex.TARGET_TEXT_PROJ,
    ex.TARGET_CAPTION_PROJ,
    ex.TARGET_SPEAKER,
    ex.TARGET_DURATION,
    ex.TARGET_DIT,
    ROLE_CODEC,
)

#: latent（z）を動かしうる役割。`duration` は S にしか効かず、`codec` は latent の後段なので、
#: どちらもグリッドを固定した比較では z をビット単位で動かさない（モジュール docstring の
#: 「直交性の門」）。
LATENT_ROLES: frozenset[str] = frozenset(ROLES) - {ex.TARGET_DURATION, ROLE_CODEC}

#: **ケースに参照があるときだけ** z に効く役割。参照なしのケースでは speaker encoder が
#: 走らない（ホストのゼロ供給短絡 — ADR 0048）ので、`speaker` を丸めても z は不変が正しい
#: （実測 2026-08-12: `i8-speaker-only` の no-ref は z も波形も基準とビット一致）。
REFERENCE_ONLY_ROLES: frozenset[str] = frozenset({ex.TARGET_SPEAKER})

#: ケース名 → 参照音声の有無（直交性の門がケース条件で期待を割るための表）。
CASE_HAS_REFERENCE: Mapping[str, bool] = MappingProxyType(
    {case.name: case.reference is not None for case in ip.PIPELINE_CASES}
)


# ---- w4 の方式（スクリーニングの勝者 4 種）----------------------------------

#: w4 の group 長 = **32 固定**（core の既定 = ADR 0069 追記 1 で確定した値）。方式比較で g を
#: 同時に振らないのは、方式の差と g の差が混ざると「どちらが効いたか」が言えなくなるため
#: （g 軸の再評価は方式確定後 — ADR 0069 追記 5 の 3）。
W4_GROUP_SIZE = DEFAULT_GROUP_SIZE

#: w4 の丸め対象 op 種 = i8 と同じ 5 種（`fake_quant_int4` の `op_types` 明示 opt-in —
#: ADR 0069 追記 5 の 1）。**出荷経路ではない** — i4 の実行経路は linear 限定（決定 5）で、
#: conv 系の scale は受容野平坦化の rank 2 になり emit へ構造的に渡せない。ここで広げるのは
#: 「非 linear まで丸めたときに品質がどこまで戻るか」を runtime 非接触で測る側。
W4_OP_TYPES: tuple[type[nn.Module], ...] = QUANT_MODULE_TYPES

#: k-means の表 1 枚のビット数（16 centroid × f32）。
CODEBOOK_BITS = DEFAULT_CODEBOOK_LEVELS * 32


class TargetCounts(NamedTuple):
    """w4 対象テンソル集合の計数（サイズ試算の入力）。

    `channels` は `karume.quantize.channel_rows` の行数の総和、`modules` は層数。
    """

    modules: int
    channels: int
    elements: int

    @property
    def groups(self) -> int:
        """g32 group の総数（group ごとに scale 1 個が要る方式のサイズ試算に使う）。"""
        return self.elements // W4_GROUP_SIZE


def total_counts(counts: Sequence[TargetCounts]) -> TargetCounts:
    """役割ごとの計数を 1 本へ足す（サイズ試算はモデル全体で読む）。"""
    return TargetCounts(
        modules=sum(item.modules for item in counts),
        channels=sum(item.channels for item in counts),
        elements=sum(item.elements for item in counts),
    )


class Excluded(NamedTuple):
    """量子化軸が g32 で割り切れず対象から外した重み（一覧は出力へ必ず載せる）。"""

    module: str
    fqn: str
    axis_length: int
    elements: int


def group_axis_length(weight: torch.Tensor, axis: int) -> int:
    """`karume.quantize.channel_rows` で畳んだときの量子化軸長（= 要素数 / チャネル数）。

    実際に `channel_rows` を呼ばないのは `ConvTranspose1d` の平坦化が `reshape` の**コピー**
    になるため（形だけ知りたい計数で重み 1 本ぶんの一時領域を作らない）。
    """
    return weight.numel() // int(weight.shape[axis])


class RoleScan(NamedTuple):
    """1 つの木の w4 対象規模（g32 に載った側と、割り切れず外した側）。"""

    counts: TargetCounts
    excluded: tuple[Excluded, ...]


def scan_targets(model: nn.Module, op_types: tuple[type[nn.Module], ...]) -> RoleScan:
    """`iter_quant_targets` の対象を「g32 に載る側 / 割り切れず外す側」へ割る。

    対象選択は core（格納経路と同じ `iter_quant_targets`）を通す — 写した別実装にすると
    「測った対象」と「丸めた対象」が黙って割れる。端数 group を作らないのは格納側の制約
    そのもの（ADR 0069 決定 2）なので、割り切れない重みは丸めずに**一覧で出す**。
    """
    modules = channels = elements = 0
    excluded: list[Excluded] = []
    for fqn, weight, axis in iter_quant_targets(model, op_types):
        length = group_axis_length(weight, axis)
        if length % W4_GROUP_SIZE:
            excluded.append(Excluded(fqn.removesuffix(".weight"), fqn, length, weight.numel()))
            continue
        modules += 1
        channels += int(weight.shape[axis])
        elements += weight.numel()
    return RoleScan(TargetCounts(modules, channels, elements), tuple(excluded))


def aligned_include(excluded: Sequence[Excluded]) -> Callable[[str], bool]:
    """g32 で割り切れないモジュールを落とす `include` 述語（モジュール FQN で引く）。"""
    dropped = {item.module for item in excluded}
    return lambda name: name not in dropped


@dataclass(frozen=True)
class W4Method:
    """w4 の丸め方式 1 種 — 名前・丸めの当て方・サイズ試算の式を 1 行に束ねる。

    3 つを別表に散らすと「品質を測った方式」と「サイズを試算した方式」が黙って割れる
    （方式を 1 種足したときに片方だけ更新される形になる）。丸めの実装は core の共有
    （`karume.quantize` / `karume.quant_methods`）で、ここが持つのは呼び分けと式だけ。
    """

    #: スクリーニングと同じ方式名（表・JSON にそのまま出る）。
    name: str
    #: 丸めを model へ in-place で当てる（`(model, op_types, include, fit_stride)` →
    #: `describe()` を持つ報告）。`op_types` は構成側（{@link Recipe.op_types}）が決める —
    #: ここへ焼くと「数えた対象」と「丸めた対象」が構成ごとに割れうる。`fit_stride` を
    #: 読むのは {@link W4_KMEANS_SHARED} だけ。
    round_model: Callable[[nn.Module, tuple[type[nn.Module], ...], Callable[[str], bool], int], Any]
    #: 対象集合の投影ビット数（`(計数, 表の枚数)`）。
    projected_bits: Callable[[TargetCounts, int], float]
    #: `projected_bits` の式（**出力へそのまま載せる** — 投影の前提を表から追えるように）。
    formula: str
    #: 表の fit を部分標本にできるか（`--kmeans-fit-stride` が効く方式か）。
    samples_fit: bool = False


#: RTN（= 格納形 `i4`・ADR 0069 決定 3）。w4 列の**基準**で、他の 3 方式はこれとの差で読む。
W4_RTN = W4Method(
    "rtn-i4-g32",
    lambda model, op_types, include, _stride: fake_quant_int4(
        model, W4_GROUP_SIZE, include=include, op_types=op_types
    ),
    lambda counts, _tables: 4 * counts.elements + 32 * counts.groups,
    "4bit + g32 f32 scale = 5.0 bpw",
)

W4_NF4 = W4Method(
    "nf4",
    lambda model, op_types, include, _stride: fake_quant_nf4(
        model, W4_GROUP_SIZE, include=include, op_types=op_types
    ),
    lambda counts, _tables: 4 * counts.elements + 32 * counts.groups,
    "4bit + g32 f32 scale = 5.0 bpw",
)

W4_MXFP4 = W4Method(
    "mxfp4",
    lambda model, op_types, include, _stride: fake_quant_mxfp4(
        model, W4_GROUP_SIZE, include=include, op_types=op_types
    ),
    lambda counts, _tables: 4 * counts.elements + 8 * counts.groups,
    "4bit + g32 E8M0 scale = 4.25 bpw",
)

#: k-means の共有表。**表は役割ごとに 1 枚**（`tables` = 対象を持つ役割数）— 配布は役割ごとに
#: 別の IR 成果物なので、成果物をまたいで 1 枚の表を共有する席が格納側に無い。丸めも役割
#: ごとに閉じて当てる（`apply_weight_quant`）ので、試算と実際の表の枚数が一致する。
#:
#: NOTE: `shared` は表を張る前に対象の正規化値を 1 本へ連結するので、dit 役割（丸める木で
#: 7.5 億要素）では f64 の作業領域まで含めて 20GB 超になる。`--kmeans-fit-stride` が core の
#: 逃げ道（**表の fit だけ**等間隔部分標本・適用は常に全量）。
W4_KMEANS_SHARED = W4Method(
    "kmeans:shared",
    lambda model, op_types, include, stride: fake_quant_kmeans(
        model, "shared", W4_GROUP_SIZE, include=include, op_types=op_types, fit_stride=stride
    ),
    lambda counts, tables: 4 * counts.elements + 32 * counts.groups + CODEBOOK_BITS * tables,
    "4bit + g32 f32 scale + 表 16×f32 を役割ごとに 1 枚",
    samples_fit=True,
)


# ---- 校正付き丸めの構成（波 J-2）--------------------------------------------

#: 校正付き構成の対象名（JSON の `w4.opTypes` と並ぶ「何を丸めたか」の綴り）。w4 方式の
#: `i4-linear`（全役割の linear）とは**集合が違う** — 校正の駆動は stage 内の `nn.Linear` に
#: 閉じるので、DiT block の外（`in_proj` / `out_proj` / `cond_module`）も他役割も入らない。
CALIB_TARGET = "dit:blocks-linear"

#: DiT block 列の**モデル内 FQN 接頭辞**（`load_dit` が組む `TextToLatentRFDiT` の `.blocks`）。
#: scale 台帳のキーを `Int4Report` と同じ FQN 空間へ揃えるために stage へ渡す
#: （core の `StageSpec` の契約）。
DIT_BLOCK_PREFIX = "blocks"


@dataclass(frozen=True)
class CalibConfig:
    """校正付き丸め 1 本ぶんの指定（方式 × 格納グリッド）。

    `projected_bits` / `formula` は {@link W4Method} と同じ器で、{@link project_size} が
    両方を受ける — 校正は**格子を 1 バイトも変えない**（同じ格子の中で丸め先を選び直すだけ）
    ので、格納形の式は方式側と共有できる。
    """

    #: 構成名（`--config` の綴り・表と JSON にそのまま出る）。
    name: str
    #: core の校正方式（`karume.quant_calib`）。
    method: CalibMethod
    #: 丸め先の格納グリッド。
    grid: GridSpec
    #: 対象集合の投影ビット数（`(計数, 表の枚数)`）。
    projected_bits: Callable[[TargetCounts, int], float]
    #: `projected_bits` の式（**出力へそのまま載せる**）。
    formula: str


def _calib_group_scaled_bits(counts: TargetCounts, _tables: int) -> float:
    """group ごとに f32 scale 1 個を持つ格子の総 bit（`rtn` / `nf4` 共通）。"""
    return 4 * counts.elements + 32 * counts.groups


def _calib_codebook_bits(counts: TargetCounts, tables: int) -> float:
    """`kmeans_shared` の総 bit — 表は**層ごとに 1 枚**（`tables` = 対象の層数）。"""
    return _calib_group_scaled_bits(counts, tables) + CODEBOOK_BITS * tables


#: 校正付き構成 3 本（この順で表に並ぶ）。丸めは core の `karume.quant_calib` の共有で、
#: ここが持つのは呼び分けと投影式だけ。AWQ を置かないのは、等価倍率 `s` が単独で格納できず
#: （fold か companion が要る）出口の無い列になるため（core のモジュール docstring の MUST）。
CALIB_CONFIGS: tuple[CalibConfig, ...] = (
    CalibConfig(
        "gptq-rtn",
        "gptq",
        GridSpec(kind="rtn", group_size=W4_GROUP_SIZE),
        _calib_group_scaled_bits,
        "4bit + g32 f32 scale = 5.0 bpw（格納は RTN i4 そのもの — 校正は丸め先だけを変える）",
    ),
    CalibConfig(
        "gptq-nf4",
        "gptq",
        GridSpec(kind="nf4", group_size=W4_GROUP_SIZE),
        _calib_group_scaled_bits,
        "4bit + g32 f32 scale = 5.0 bpw（格子は NF4 の固定表）",
    ),
    CalibConfig(
        "gptq-kmeans",
        "gptq",
        GridSpec(kind="kmeans_shared", group_size=W4_GROUP_SIZE),
        _calib_codebook_bits,
        "4bit + g32 f32 scale + 表 16×f32 を**層ごと**に 1 枚"
        "（射程が層内 — 役割ごとに 1 枚の kmeans:shared とは別式）",
    ),
)

CALIB_NAMES: tuple[str, ...] = tuple(config.name for config in CALIB_CONFIGS)


def check_fit_stride(recipe: Recipe, fit_stride: int) -> None:
    """`--kmeans-fit-stride` が効かない構成へ渡されたら fail loudly（黙って無視しない）。"""
    if fit_stride == 1:
        return
    if recipe.calib is not None:
        raise SystemExit(
            f"--kmeans-fit-stride {fit_stride} は校正付き構成には効かない"
            f"（{recipe.weight} の表は**層ごと**に張るので全量 fit でも作業領域が小さい）"
        )
    if recipe.method is None or not recipe.method.samples_fit:
        raise SystemExit(
            f"--kmeans-fit-stride {fit_stride} は表の fit を部分標本にする方式専用"
            f"（この構成の丸めは {recipe.weight}）"
        )


@dataclass(frozen=True)
class SizeProjection:
    """方式 × 対象集合の**投影**サイズ（実測ではない — 式は {@link W4Method.formula}）。"""

    counts: TargetCounts
    tables: int
    bits: float
    formula: str

    @property
    def bits_per_weight(self) -> float:
        return self.bits / self.counts.elements

    @property
    def projected_mib(self) -> float:
        return self.bits / 8 / 1024**2

    @property
    def f32_mib(self) -> float:
        """同じ対象集合を f32 で持ったときの MiB（投影の分母 — 縮小率を表から読めるように）。"""
        return self.counts.elements * 4 / 1024**2


def project_size(
    method: W4Method | CalibConfig, counts: TargetCounts, tables: int
) -> SizeProjection:
    """方式の式を対象集合の計数へ当てる（`tables` = 表を張る成果物の数）。

    校正付き構成（{@link CalibConfig}）も同じ器で受ける — 校正は格子を 1 バイトも変えないので、
    格納形の式は方式側と共有できる。
    """
    return SizeProjection(counts, tables, method.projected_bits(counts, tables), method.formula)


@dataclass(frozen=True)
class Recipe:
    """1 構成の量子化レシピ（`weight` が `None` なら丸めない）。"""

    #: 丸めの綴り（格納 dtype `f16` / `i8`、または w4 の方式名）。
    weight: str | None
    roles: tuple[str, ...]
    #: DiT の `nn.Linear` 入力へ per-token i8 の fake-quant を掛けるか（w8a8 の活性側）。
    act_quant: bool = False
    #: w4 方式（`None` = `weight` が格納 dtype の綴りで `irodori.export.fake_quant` を通る側）。
    method: W4Method | None = None
    #: w4 の丸め対象 op 種（`method` があるときだけ意味を持つ）。既定は測定用に広げた 5 種で、
    #: `(nn.Linear,)` に絞った形が**今日の配布対応形**（i4 の実行経路 — ADR 0069 決定 5）。
    op_types: tuple[type[nn.Module], ...] = W4_OP_TYPES
    #: 校正付き丸め（`None` = 重みだけを見て丸める側）。`method` とは**排他** — 校正は
    #: stage 逐次で駆動するので、`apply_weight_quant` の役割一括の経路を通らない。
    calib: CalibConfig | None = None


def w4_recipe(
    method: W4Method,
    roles: tuple[str, ...],
    op_types: tuple[type[nn.Module], ...] = W4_OP_TYPES,
) -> Recipe:
    """w4 方式 1 本ぶんのレシピ（`weight` の綴りは方式名 — 二重管理を作らない）。"""
    return Recipe(method.name, roles, method=method, op_types=op_types)


#: 活性シムの比較相手（**同じ重みで活性だけ素の**構成）。素通り検出はこの 1 本との差で見る。
WEIGHT_ONLY_BASE = "i8-all"

#: 混成形の役割（duration だけ f32 据え置き — ADR 0050 決定 6 の (i) 案）。i8 と w4 で
#: **同じ軸**にする（別々に書くと片方だけ動いたときに 2 つの混成が黙って別物になる）。
MIXED_ROLES: tuple[str, ...] = tuple(role for role in ROLES if role != ex.TARGET_DURATION)

#: 主要 5 構成（聴き比べと配布形の裁定に載る側）。
CONFIGS: Mapping[str, Recipe] = MappingProxyType(
    {
        BASE_CONFIG: Recipe(None, ()),
        "f16": Recipe("f16", ROLES),
        WEIGHT_ONLY_BASE: Recipe("i8", ROLES),
        # ADR 0050 決定 6 の (i) 案 — duration だけ f32 据え置き。
        "i8-mixed": Recipe("i8", MIXED_ROLES),
        # 配布形の `w8a8` 席 — 重みは `i8-all` と 1 バイトも変わらず、DiT の活性だけが i8。
        "w8a8": Recipe("i8", ROLES, act_quant=True),
    }
)

#: 直交分解（グラフ境界ごとに 1 本ずつ i8 にする — 劣化の帰属先を割る）。
DECOMPOSED_ROLES: tuple[str, ...] = (
    ex.TARGET_DIT,
    ex.TARGET_BACKBONE,
    ex.TARGET_SPEAKER,
    ex.TARGET_DURATION,
    ROLE_CODEC,
)
DIAGNOSTICS: Mapping[str, Recipe] = MappingProxyType(
    {f"i8-{role}-only": Recipe("i8", (role,)) for role in DECOMPOSED_ROLES}
)

#: w4 スクリーニングの勝者 4 種 × 全役割形 + RTN の duration 据え置き混成。
#:
#: 第一の門は i8 と**同じ軸** — S が動くか（ADR 0050 決定 6）。動いた場合の受け皿として
#: RTN にだけ混成形を置く（S が動くかどうかは丸めの粗さで決まるので、4 方式ぶんの混成を
#: 並べても同じ問いを 4 回聞くだけになる。RTN で受け皿の形が立てば他方式へ横展開できる）。
#:
#: 全役割形の対象 op 種は {@link W4_OP_TYPES}（linear / conv 系 / embedding）。
#:
#: `i4-linear` は**今日の配布対応形**（RTN i4 × linear 限定 = 実際に出荷できる唯一の形 —
#: ADR 0069 決定 5）。`--scan` の実測で linear が対象要素の 81.8% を覆い「conv 主体なら
#: 省略」の前提が成立しなかったため、品質構成として置く（2026-08-19 全体レビュー後の追補）。
#: roles から codec を外すのは codec に linear が 1 本も無いから（対象 0 の役割を回さない —
#: 除外ではなく恒等なので構成の意味は「全役割の linear」のまま）。linear の混成形（duration
#: 除外）は置かない — 全役割の実測で S ドリフトの原因が duration の重みでなく上流特徴の
#: 変形だと確定した（i4-mixed の強制グリッド品質が i4-all とビット一致）ため、混成しても
#: S は救えない。
W4_LINEAR_ROLES: tuple[str, ...] = tuple(role for role in ROLES if role != ROLE_CODEC)
W4_CONFIGS: Mapping[str, Recipe] = MappingProxyType(
    {
        "i4-all": w4_recipe(W4_RTN, ROLES),
        "i4-linear": w4_recipe(W4_RTN, W4_LINEAR_ROLES, op_types=(nn.Linear,)),
        "i4-mixed": w4_recipe(W4_RTN, MIXED_ROLES),
        "nf4-all": w4_recipe(W4_NF4, ROLES),
        "mxfp4-all": w4_recipe(W4_MXFP4, ROLES),
        "kmeans-shared-all": w4_recipe(W4_KMEANS_SHARED, ROLES),
    }
)

#: 校正付き構成（波 J-2）— 役割は `dit` だけ（丸めるのは DiT block 列の linear で、他役割は
#: 1 ビットも動かない）。`weight` の綴りは構成名そのもの（二重管理を作らない — {@link w4_recipe}
#: と同じ流儀）。
CALIB_RECIPES: Mapping[str, Recipe] = MappingProxyType(
    {
        config.name: Recipe(config.name, (ex.TARGET_DIT,), op_types=(nn.Linear,), calib=config)
        for config in CALIB_CONFIGS
    }
)

#: 構成名 → レシピの全表。
RECIPES: Mapping[str, Recipe] = MappingProxyType(
    {**CONFIGS, **DIAGNOSTICS, **W4_CONFIGS, **CALIB_RECIPES}
)

#: LSD の STFT 設定（48kHz の解析グリッド — hop 512 ≈ 10.7ms）。
STFT_N_FFT = 2048
STFT_HOP = 512
#: 振幅の床（参照の最大振幅に対する相対値 = −100dB）。無音区間の対数が −inf へ落ちるのを
#: 防ぐためだけの下限で、両系列に**同じ床**を当てる（片側だけに当てると差が床に依存する）。
STFT_FLOOR = 1e-5

CASE_SUFFIX = ".safetensors"
REPORT_FILE = "report.json"


# ---- 指標（`measure_quant_sbv2.py` からの複製 — モジュール docstring の NOTE）----------


def snr_db(value: torch.Tensor, reference: torch.Tensor) -> float:
    """`10·log10(Σref² / Σ(value−ref)²)`（構成間の音質比較の主指標）。"""
    err = float((value - reference).double().square().sum())
    ref = float(reference.double().square().sum())
    if err == 0.0:
        return math.inf
    if ref == 0.0:
        return -math.inf
    return 10.0 * math.log10(ref / err)


def log_spectral_distance(value: torch.Tensor, reference: torch.Tensor) -> float:
    """対数スペクトル距離 `sqrt(mean((20log10|X| − 20log10|Y|)²))` [dB]。

    波形 SNR は位相が少しずれるだけで崩れる（flow-matching の軌道が変われば、聴感上同じ音でも
    残差が信号と同オーダーになる）。LSD は振幅スペクトルだけを見るので**位相ずれに鈍く**、
    「音色・帯域の壊れ方」を測る側の指標として併記する。どちらか一方では判断しない。
    """
    window = torch.hann_window(STFT_N_FFT)
    spectra = [
        torch.stft(
            wave_data, n_fft=STFT_N_FFT, hop_length=STFT_HOP, window=window, return_complex=True
        ).abs()
        for wave_data in (value, reference)
    ]
    floor = float(spectra[1].max()) * STFT_FLOOR
    a, b = (20.0 * torch.log10(spectrum.clamp_min(floor)) for spectrum in spectra)
    return float((a - b).square().mean().sqrt())


def rel_rms(value: torch.Tensor, reference: torch.Tensor) -> float:
    denom = float(reference.double().square().sum())
    if denom == 0.0:
        return 0.0
    return math.sqrt(float((value - reference).double().square().sum()) / denom)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    """モノラル 16bit PCM で書く（`decodeWav` の逆 — 書きは ×32767 が Karume の規約）。"""
    clipped = audio.detach().reshape(-1).to(torch.float32).clamp(-1.0, 1.0)
    pcm = torch.floor(clipped * 32767.0 + 0.5).to(torch.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.numpy().tobytes())


# ---- 構成の組み立て ---------------------------------------------------------


def role_modules(modules: Mapping[str, nn.Module], roles: Sequence[str]) -> dict[str, nn.Module]:
    """役割 → その役割の丸め対象モジュール（`irodori.export.fake_quant` へ渡す形）。

    MUST: 3 つの norm を**消費側の役割**へ束ねる（`speaker_norm` は speaker グラフ・
    `text_norm` は duration グラフ・`caption_norm` は caption-proj グラフの内側 — 所在は
    ADR 0048 決定 1）。全役割を選ぶと `irodori.export.export_series` /
    `irodori.pipeline_ref.emit` が丸める 9 本と**同じ集合**になり、f16 構成が配布系列の
    対応物であることが集合として保証される。
    """
    grouped: dict[str, dict[str, nn.Module]] = {
        ex.TARGET_BACKBONE: {ex.TARGET_BACKBONE: modules[ex.TARGET_BACKBONE]},
        ex.TARGET_TEXT_PROJ: {ex.TARGET_TEXT_PROJ: modules[ex.TARGET_TEXT_PROJ]},
        ex.TARGET_CAPTION_PROJ: {
            ex.TARGET_CAPTION_PROJ: modules[ex.TARGET_CAPTION_PROJ],
            "caption_norm": modules["caption_norm"],
        },
        ex.TARGET_SPEAKER: {
            ex.TARGET_SPEAKER: modules[ex.TARGET_SPEAKER],
            "speaker_norm": modules["speaker_norm"],
        },
        ex.TARGET_DURATION: {
            ex.TARGET_DURATION: modules[ex.TARGET_DURATION],
            "text_norm": modules["text_norm"],
        },
        ex.TARGET_DIT: {ex.TARGET_DIT: modules[ex.TARGET_DIT]},
    }
    picked: dict[str, nn.Module] = {}
    for role in roles:
        picked.update(grouped.get(role, {}))
    return picked


def apply_weight_quant(
    recipe: Recipe, scoped: Mapping[str, nn.Module], fit_stride: int = 1
) -> dict[str, str]:
    """構成ぶんの重みの丸めを役割モジュールへ当て、役割 → 要約を返す。

    f16 / i8 はこれまでどおり `irodori.export.fake_quant`（配布経路と同じ関数）へ委ねる。
    w4 は方式ごとに core の丸めを呼び、量子化軸が g32 で割り切れない重みだけ `include` で
    落とす（端数 group を作らない — ADR 0069 決定 2）。

    MUST: 丸めの対象は w4 でも**役割モジュールの木**（= i8 と同じ集合）にする。配布グラフに
    載る部分木だけ丸めると i8 の測定値と「何を丸めたか」が食い違い、方式間の比較が成立
    しなくなる（グラフに載らない内側のコピーまで丸まるが、forward に出てこないので品質には
    効かない）。サイズ試算の計数だけは配布グラフ側から採る（{@link graph_scans}）。

    MUST: 役割に w4 の候補があるのに全部が g32 で落ちたら fail loudly — その役割だけ f32 の
    まま回すと「丸めたつもりの構成」を測ってしまう。
    """
    if recipe.method is None:
        return dict(ex.fake_quant(recipe.weight, scoped).reports)
    reports: dict[str, str] = {}
    for name, module in sorted(scoped.items()):
        scan = scan_targets(module, recipe.op_types)
        if scan.counts.modules == 0:
            if scan.excluded:
                raise SystemExit(
                    f"{name}: w4 の対象 {len(scan.excluded)} 本が全て量子化軸を"
                    f" g32 で割り切れず 0 本になった（{scan.excluded[0].fqn} の軸長"
                    f" {scan.excluded[0].axis_length} 他）"
                )
            reports[name] = "格納 f32 のまま（w4 の対象型を持たない）"
            continue
        rounded = recipe.method.round_model(
            module, recipe.op_types, aligned_include(scan.excluded), fit_stride
        )
        if rounded.modules != scan.counts.modules:
            raise SystemExit(
                f"{name}: 丸めた本数 {rounded.modules} が対象 {scan.counts.modules} と違う"
                "（op_types か include が scan と割れている — 数えた対象と丸めた対象は"
                "同じ集合でなければならない）"
            )
        skipped = f" / g32 非整列 {len(scan.excluded)} 本を除外" if scan.excluded else ""
        sampled = f" / 表の fit は 1/{fit_stride} 部分標本" if fit_stride > 1 else ""
        reports[name] = f"{recipe.method.name} へ丸めた — {rounded.describe()}{skipped}{sampled}"
    for name, report in reports.items():
        print(f"[fake-quant] {name}: {report}", flush=True)
    return reports


def role_graphs(graphs: ip.HostGraphs) -> dict[str, nn.Module]:
    """役割 → **配布グラフ**（`irodori.export.export_series` が emit する 6 本のラッパ）。"""
    return {
        ex.TARGET_BACKBONE: graphs.backbone,
        ex.TARGET_TEXT_PROJ: graphs.text_proj,
        ex.TARGET_CAPTION_PROJ: graphs.caption_proj,
        ex.TARGET_SPEAKER: graphs.speaker,
        ex.TARGET_DURATION: graphs.duration,
        ex.TARGET_DIT: graphs.dit,
    }


def graph_scans(
    graphs: Mapping[str, nn.Module],
    roles: Sequence[str],
    op_types: tuple[type[nn.Module], ...],
) -> dict[str, RoleScan]:
    """役割 → 配布グラフに載る w4 対象の規模（サイズ試算と除外一覧の入力）。

    丸めた木ではなくラッパを数えるのは、`irodori.export.load_dit` が組む DiT が**グラフに
    載らない backbone のコピー**まで抱えるため（`export.TARGET_SCALE_SOURCES` の NOTE）。
    丸めた集合で試算すると、出荷しない重みのぶんだけ投影 MiB が膨らむ。op 種は丸めと同じ
    ものを受ける（別々に動くと「丸めた対象」と「試算した対象」が黙って割れる）。
    """
    return {
        role: scan_targets(graph, op_types) for role, graph in graphs.items() if role in set(roles)
    }


class Loaded(NamedTuple):
    """`load_modules` の戻り（丸めを当てる前の素の一式）。"""

    source: Any
    modules: dict[str, nn.Module]
    config: Any
    text_config: Mapping[str, Any]
    model_config: Mapping[str, Any]


def load_modules(model_dir: Path, source_dir: Path) -> Loaded:
    """`irodori.pipeline_ref.emit` と同じ 9 本を組む（丸めはまだ当てない）。"""
    source = ex.IrodoriSource(source_dir)
    text_config, model_config = ex.read_configs(model_dir)
    state = load_file(str(model_dir / ex.MODEL_FILE))
    backbone = ex.load_backbone(source, state, text_config)
    hidden_size = int(backbone.hidden_size)
    config = source.model_config(model_config)
    modules = {
        ex.TARGET_BACKBONE: backbone,
        ex.TARGET_TEXT_PROJ: ex.load_projector(
            source,
            state,
            model_config,
            ex.TEXT_PROJ_PREFIX,
            hidden_size,
            int(model_config["text_dim"]),
        ),
        ex.TARGET_CAPTION_PROJ: ex.load_projector(
            source,
            state,
            model_config,
            ex.CAPTION_PROJ_PREFIX,
            hidden_size,
            int(model_config["caption_dim"]),
        ),
        ex.TARGET_SPEAKER: ex.load_speaker_encoder(source, state, config),
        ex.TARGET_DURATION: ex.load_duration_predictor(source, state, config),
        ex.TARGET_DIT: ex.load_dit(source, state, config, text_config),
        "speaker_norm": ex.load_rms_norm(
            source, state, ex.SPEAKER_NORM_PREFIX, int(config.speaker_dim), float(config.norm_eps)
        ),
        "text_norm": ex.load_rms_norm(
            source, state, ex.TEXT_NORM_PREFIX, int(config.text_dim), float(config.norm_eps)
        ),
        "caption_norm": ex.load_rms_norm(
            source,
            state,
            ex.CAPTION_NORM_PREFIX,
            int(config.caption_dim_resolved),
            float(config.norm_eps),
        ),
    }
    return Loaded(source, modules, config, text_config, model_config)


def build_graphs(
    modules: Mapping[str, nn.Module], config: Any, model_config: Mapping[str, Any]
) -> ip.HostGraphs:
    """`irodori.pipeline_ref.emit` と同じ 6 本のグラフラッパ。

    MUST: **パッチ適用後**でしか正しく動かない（実数形 RoPE 表を渡すため）。
    """
    speaker_max = ex.speaker_sym_max(model_config)
    return ip.HostGraphs(
        backbone=ex.BackboneGraph(modules[ex.TARGET_BACKBONE]),
        text_proj=ex.ProjectorGraph(modules[ex.TARGET_TEXT_PROJ]),
        caption_proj=ex.CaptionProjectorGraph(
            modules[ex.TARGET_CAPTION_PROJ], modules["caption_norm"]
        ),
        speaker=ex.SpeakerGraph(modules[ex.TARGET_SPEAKER], modules["speaker_norm"], speaker_max),
        duration=ex.DurationGraph(modules[ex.TARGET_DURATION], modules["text_norm"]),
        dit=ex.DitGraph(modules[ex.TARGET_DIT], ex.dit_sym_max(config)),
    )


class DecoderStage(NamedTuple):
    """{@link build_decoder} の戻り。`scan` は w4 構成でコーデックを丸めたときだけ埋まる。"""

    graph: nn.Module
    sample_rate: int
    report: str | None
    scan: RoleScan | None


def build_decoder(
    source_dir: Path, model_dir: Path, recipe: Recipe, fit_stride: int = 1
) -> DecoderStage:
    """コーデックの decode 経路（`DecoderGraph`）とサンプリング周波数・丸めの要約を返す。

    MUST（順序）: 丸めは `fold_weight_norm` / `lift_snake_alphas` の**後**（`irodori.dacvae.export`
    の `_fake_quant` の順序 MUST — ここは同じ前処理を同じ順で通してから丸めを当てる）。
    切り詰めた `in_proj` は decode 経路に出てこないので組まない。

    w4 は `dv._fake_quant`（格納 dtype の綴りしか受けない）を通らず、方式の丸めを**コーデック
    モデル全体**へ当てる。計数と除外一覧は decode 経路のラッパから採る（{@link graph_scans}
    と同じ理由 — encoder は配布されない）。

    NOTE: `DecoderGraph` の木には透かし枝の未使用層が残る（`bypass_watermark` は forward を
    差し替えるだけでモジュールは外さない）ので、計数は decode が実際に通る集合より
    **6,266,080 要素（24MiB 相当・コーデックの 8.8%）多い**。`dv.main_path` で正確に割れるが、
    あちらは「門でのみ使う写し」なので計数へは持ち込まない（モデル全体では 0.8% の上振れ）。
    """
    source = dv.DacvaeSource(source_dir)
    model = dv.load_codec(source, model_dir)
    dv.bypass_watermark(model.decoder)
    dv.fold_weight_norm(model)
    dv.lift_snake_alphas(source, model)
    quantize = ROLE_CODEC in recipe.roles and recipe.weight is not None
    if not quantize:
        report = dv._fake_quant("f32", model).report
    elif recipe.method is None:
        report = dv._fake_quant(recipe.weight, model).report
    else:
        scan = scan_targets(model, recipe.op_types)
        if scan.counts.modules == 0:
            raise SystemExit(
                f"codec: w4 の対象 {len(scan.excluded)} 本が全て量子化軸を g32 で"
                "割り切れず 0 本になった"
            )
        rounded = recipe.method.round_model(
            model, recipe.op_types, aligned_include(scan.excluded), fit_stride
        )
        sampled = f" / 表の fit は 1/{fit_stride} 部分標本" if fit_stride > 1 else ""
        report = f"{recipe.method.name} へ丸めた — {rounded.describe()}{sampled}"
        print(f"[fake-quant] codec: {report}", flush=True)
    graph = dv.DecoderGraph(model.quantizer.out_proj, model.decoder)
    return DecoderStage(
        graph,
        int(model.sample_rate),
        report,
        scan_targets(graph, recipe.op_types) if quantize and recipe.method is not None else None,
    )


# ---- 校正付き丸めの駆動（stage 分解 + Catcher）-------------------------------


class _CalibStepsReached(Exception):  # noqa: N818 — 異常ではなく打ち切りの合図
    """必要な step 数ぶん捕まえた合図（参照ループを途中で畳むための番兵）。"""


class DitBlockStage(nn.Module):
    """DiT block 1 枚を「hidden を位置引数で受ける」形へ包む stage ラッパ。

    block 本体は `irodori.export.DitGraph.forward` のループ 4 行と**同じ順序**で組む —
    `TextToLatentRFDiT` の block は `DiffusionBlock.forward` を持つが、`DitGraph` は
    それを呼ばずに attention の同値実装（{@link irodori.export.DitGraph._attention}）へ
    展開するので、block をそのまま stage にすると**測っている経路が別物**になる。attention は
    その staticmethod を直に借りて写しを増やさない。

    MUST: 写しが上流とずれていないことは {@link assert_stage_split_matches_graph} が丸める前に
    ビット一致で実測する（`nn.Dropout(p=0.0)` を落とすのは `DitGraph` と同じ厳密恒等）。

    子モジュールの名前を**block 番号**にしてあるのは、stage 内の局所 FQN
    （`0.attention.wq.weight`）へ {@link DIT_BLOCK_PREFIX} を足すだけでモデル内 FQN へ
    戻すため。
    """

    def __init__(self, index: int, block: nn.Module) -> None:
        super().__init__()
        self.child = str(index)
        self.add_module(self.child, block)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text: torch.Tensor,
        speaker: torch.Tensor,
        caption: torch.Tensor,
        mask: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        block = getattr(self, self.child)
        h, attention_gate = block.attention_adaln(x, cond_embed)
        x = x + attention_gate * ex.DitGraph._attention(
            block.attention, h, text, speaker, caption, mask, freqs
        )
        h, mlp_gate = block.mlp_adaln(x, cond_embed)
        return x + mlp_gate * block.mlp(h)


def dit_stages(graph: nn.Module) -> tuple[StageSpec, ...]:
    """実行順の DiT block を `(モデル内 FQN 接頭辞, stage)` で返す。"""
    return tuple(
        (DIT_BLOCK_PREFIX, DitBlockStage(index, block)) for index, block in enumerate(graph.blocks)
    )


def calib_targets(stages: Sequence[StageSpec]) -> tuple[dict[str, torch.Tensor], TargetCounts]:
    """stage 内の校正対象を fqn 引きの重みと計数で返す（**走査** = 過不足一致門の基準）。

    対象選択は core の `iter_quant_targets` の共有（{@link scan_targets} と同文）。

    MUST: g32 非整列は**除外せず fail loudly** — 校正は stage を丸ごと駆動する形なので、
    途中の 1 本だけ `include` で外すと「走査の本数 = 丸めた本数」の門が張れなくなる
    （w4 方式側は層ごとに独立なので除外一覧を出す運用でよい）。
    """
    weights: dict[str, torch.Tensor] = {}
    modules = channels = elements = 0
    for prefix, stage in stages:
        for local, weight, axis in iter_quant_targets(stage, (nn.Linear,)):
            fqn = f"{prefix}.{local}"
            length = group_axis_length(weight, axis)
            if length % W4_GROUP_SIZE:
                raise SystemExit(
                    f"{fqn}: 量子化軸 {length} が g{W4_GROUP_SIZE} で割り切れない"
                    "（校正は stage 単位で駆動するので 1 本だけ外す逃げ道が無い）"
                )
            weights[fqn] = weight
            modules += 1
            channels += int(weight.shape[axis])
            elements += int(weight.numel())
    if not modules:
        raise SystemExit(f"校正対象 '{CALIB_TARGET}' に量子化できる重みが 1 本も無い")
    return weights, TargetCounts(modules, channels, elements)


def capture_case_batches(
    graph: nn.Module, run_reference: Callable[[], object], limit: int
) -> tuple[list[StageBatch], tuple[torch.Tensor, ...]]:
    """1 ケースの参照ループ（f32・CFG 込み）から **step ごとに 1 バッチ**捕まえる。

    戻りは `(バッチ列, 先頭 forward の DitGraph 引数)`。後者は
    {@link assert_stage_split_matches_graph} が stage 分解の同値を実測するのに使う。

    1 step は cond 1 回 + CFG の uncond n 回の forward になるが、その全てが**同じ `x_t`**
    を受ける（`irodori.pipeline_ref._euler` は step ごとに 1 本の `x_t` しか作らない）ので、
    `x_t` の**同一性**で step 境界を割り、新しくなった直後の 1 forward = cond 側だけを採る。

    捕まえるのは `DitGraph` が block ループの**前**に作る一式（`cond_embed` / 正規化済みの
    text・caption / speaker state / 連結マスク / RoPE 表）と、先頭 block への hidden。
    自前で組み直さないのは、組み直した瞬間に `DitGraph` の綴りと黙って割れうるから
    （EG の Catcher と同じ規律）。

    `limit` を捕まえ切ったら番兵で参照ループを畳む — `run_case` は CFG ありのループの**後**に
    CFG 無しのループをもう 1 周回すので、畳まないと「同じ step の別軌道」まで混ざる。
    """
    batches: list[StageBatch] = []
    probe: list[tuple[torch.Tensor, ...]] = []
    state: dict[str, Any] = {"x_t": None, "take": False}
    pending: dict[str, torch.Tensor] = {}

    def on_graph(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if args[0] is state["x_t"]:
            state["take"] = False
            return
        if len(batches) >= limit:
            raise _CalibStepsReached
        state["x_t"] = args[0]
        state["take"] = True
        pending["mask"] = args[2]
        pending["speaker"] = args[4]
        if not probe:
            probe.append(tuple(args))

    def on_text(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        if state["take"]:
            pending["text"] = output.detach()

    def on_caption(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        if state["take"]:
            pending["caption"] = output.detach()

    def on_block(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not state["take"]:
            return
        hidden = args[0].detach()
        batches.append(
            (
                (hidden,),
                {
                    "cond_embed": args[1].detach(),
                    "text": pending["text"],
                    "speaker": pending["speaker"],
                    "caption": pending["caption"],
                    "mask": pending["mask"],
                    "freqs": graph.rope_table[: int(hidden.shape[1])],
                },
            )
        )
        state["take"] = False

    handles = [
        graph.register_forward_pre_hook(on_graph),
        graph.text_norm.register_forward_hook(on_text),
        graph.caption_norm.register_forward_hook(on_caption),
        graph.blocks[0].attention_adaln.register_forward_pre_hook(on_block),
    ]
    try:
        run_reference()
    except _CalibStepsReached:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if not batches or not probe:
        raise SystemExit(
            "校正入力を 1 step も捕まえられなかった（DitGraph の綴りが台本の想定と食い違っている）"
        )
    return batches, probe[0]


def assert_stage_split_matches_graph(
    graph: nn.Module,
    probe: tuple[torch.Tensor, ...],
    batch: StageBatch,
    stages: Sequence[StageSpec],
) -> None:
    """stage 分解 + 尾（`out_norm` → `out_proj`）が `DitGraph` の 1 forward と**ビット一致**。

    `probe` はその forward の `DitGraph` 引数、`batch` は**同じ forward**で捕まえた先頭
    stage への入力。

    MUST: 丸める前に実測する。{@link DitBlockStage} は block ループの写しなので、上流の
    `DitGraph.forward` が変わると黙ってずれる — ずれた側で丸めると「別の経路の GPTQ」を
    測っていることになり、しかも数値は普通に出る（表からは読めない）。
    """
    args, kwargs = batch
    with torch.no_grad():
        reference = graph(*probe)
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = stage(hidden, **kwargs)
        rebuilt = graph.out_proj(graph.out_norm(hidden))
    if not torch.equal(rebuilt, reference):
        raise SystemExit(
            "stage 分解の再構成が DitGraph の出力とビット一致しない"
            f"（最大絶対差 {float((rebuilt - reference).abs().max()):.4e}）"
            " — DitBlockStage の写しが irodori.export.DitGraph.forward とずれている"
        )


@dataclass(frozen=True)
class CalibRig:
    """校正付き構成が共有する足場（stage 列・走査・先頭 stage への入力）。

    構成ごとに作り直さない — 校正入力は**丸めを 1 本も当てる前**に 1 回だけ採る（`restore`
    に当たる pristine の作法は「1 実行 = 1 構成」で担保されている）。
    """

    stages: tuple[StageSpec, ...]
    scan: Mapping[str, torch.Tensor]
    counts: TargetCounts
    batches: tuple[StageBatch, ...]
    #: ケース名 → 捕まえた step 数（縮小実行を数値の横に残す）。
    steps: Mapping[str, int]
    #: 校正に積んだ hidden の総トークン数（= Σ step ごとの S）。
    tokens: int


def build_calib_rig(
    graphs: ip.HostGraphs, run_reference: Callable[[ip.PipelineCase], object], limit: int
) -> CalibRig:
    """校正の足場を組む（stage 分解 → 走査 → 参照ループ捕捉 → 同値門）。"""
    graph = graphs.dit
    stages = dit_stages(graph)
    scan, counts = calib_targets(stages)
    batches: list[StageBatch] = []
    steps: dict[str, int] = {}
    for case in ip.PIPELINE_CASES:
        caught, probe = capture_case_batches(graph, lambda case=case: run_reference(case), limit)
        if not batches:
            # `probe` と `caught[0]` は**同じ forward**（先頭 step の cond 側）。
            assert_stage_split_matches_graph(graph, probe, caught[0], stages)
        steps[case.name] = len(caught)
        batches += caught
        print(f"[calib] {case.name}: 校正 {len(caught)} step を捕捉", flush=True)
    tokens = sum(int(args[0].shape[1]) for args, _kwargs in batches)
    print(
        f"[calib] stage {len(stages)} 段 / 対象 linear {counts.modules} 本"
        f" / バッチ {len(batches)} 本 / hidden 合計 {tokens:,} token"
        f"（対象は {CALIB_TARGET} — DiT block の外と他役割は含まない）",
        flush=True,
    )
    return CalibRig(
        stages=stages,
        scan=scan,
        counts=counts,
        batches=tuple(batches),
        steps=steps,
        tokens=tokens,
    )


def apply_calib(config: CalibConfig, rig: CalibRig) -> tuple[dict[str, str], CalibReport]:
    """校正付き丸め 1 本を DiT へ in-place で当てる（stage 逐次の駆動は core 側）。"""
    report = calibrate_stages(rig.stages, rig.batches, method=config.method, spec=config.grid)
    assert_calib_covers_scan(report, rig.scan, config.name)
    summary = f"{config.name} へ校正付きで丸めた — {report.describe()}"
    print(f"[fake-quant] {ex.TARGET_DIT}: {summary}", flush=True)
    return {ex.TARGET_DIT: summary}, report


def assert_calib_covers_scan(
    report: CalibReport, scan: Mapping[str, torch.Tensor], name: str
) -> None:
    """校正が丸めた層が stage の走査と**過不足なく**一致することを見る。

    MUST: fail loudly。stage の綴りや対象型が変わって block の一部が校正に載らなくなっても
    表には行が残り、しかも丸め漏れのぶん品質は**良い側**に出る（素通りを数字から読めない）。
    """
    rounded = {layer.fqn for layer in report.layers}
    missing = sorted(set(scan) - rounded)
    extra = sorted(rounded - set(scan))
    if missing or extra or report.modules != len(scan):
        raise SystemExit(
            f"[{name}] 校正が丸めた {report.modules} 本が走査の {len(scan)} 本と一致しない"
            f"（丸め漏れ {missing[:3]} / 走査に無い {extra[:3]}）"
        )


# ---- 1 構成の実行 ------------------------------------------------------------


def golden_z(golden_dir: Path, case: str) -> torch.Tensor:
    """full-loop golden の最終 z（`f32` 構成の恒真化を遮断する唯一の突合先）。"""
    path = golden_dir / f"{ip.CASE_PREFIX}{case}{ip.CASE_SUFFIX}"
    if not path.is_file():
        raise SystemExit(
            f"full-loop golden が無い: {path}"
            "（`uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref` で作る）"
        )
    return load_file(str(path))["z"].to(torch.float32)


def base_payload(out_dir: Path, case: str) -> dict[str, torch.Tensor]:
    """基準構成（`f32`）の z と波形。"""
    path = out_dir / f"{BASE_CONFIG}.{case}{CASE_SUFFIX}"
    if not path.is_file():
        raise SystemExit(
            f"基準構成の成果物が無い: {path}"
            f"（先に `--config {BASE_CONFIG}` を回す — 比較の基準は f32）"
        )
    return load_file(str(path))


def weight_only_path(out_dir: Path, case: str) -> Path:
    """活性シムの比較相手（{@link WEIGHT_ONLY_BASE} — 同じ重み・活性は素）の成果物。

    存在検査だけを分けてあるのは、full-loop を回す**前**に確かめるため（回し切ってから
    「比較相手が無い」で落ちると数十分ぶんの計算が捨てになる）。
    """
    path = out_dir / f"{WEIGHT_ONLY_BASE}.{case}{CASE_SUFFIX}"
    if not path.is_file():
        raise SystemExit(
            f"活性シムの比較相手が無い: {path}"
            f"（先に `--config {WEIGHT_ONLY_BASE}` を回す — 素通り検出は重みだけの構成との差）"
        )
    return path


def weight_only_payload(out_dir: Path, case: str) -> dict[str, torch.Tensor]:
    """{@link weight_only_path} の z と波形。"""
    return load_file(str(weight_only_path(out_dir, case)))


class LatentResult(NamedTuple):
    """{@link latent_stage} の戻り。`act_quant_linears` は活性シムを掛けた `nn.Linear` の本数。"""

    metas: dict[str, dict[str, Any]]
    latents: dict[str, torch.Tensor]
    reports: dict[str, str]
    act_quant_linears: int
    #: 役割 → 配布グラフ側の w4 対象規模（w4 構成のみ・他は空）。
    scans: dict[str, RoleScan]
    #: 校正付き構成の足場と丸めの計数（校正付き構成のみ・他は `None`）。
    calib: tuple[CalibRig, CalibReport] | None = None


def attach_dit_act_quant(
    recipe: Recipe, modules: Mapping[str, nn.Module]
) -> tuple[list[object], int]:
    """w8a8 構成で DiT の `nn.Linear` 入力へ per-token i8 の fake-quant を掛ける。

    MUST: **0 本は fail loudly** — 掛からないまま回すと `i8-all` と同じ数が `w8a8` の名前で
    レポートに載り、しかも品質は「良い」側に出る（活性量子化は誤差を増やすので、素通りは
    常に緑寄りの嘘になる）。掛ける先を DiT に限るのは配布形の席と揃えるため（`session` は
    `dit` の Session にだけ降りる）。
    """
    if not recipe.act_quant:
        return [], 0
    handles, attached = act_quant.attach_act_quant(modules[ex.TARGET_DIT])
    if attached == 0:
        act_quant.detach_act_quant(handles)
        raise SystemExit(
            "活性量子化シムを掛けた nn.Linear が 0 本"
            f"（`{ex.TARGET_DIT}` の Linear が適格条件 k % {act_quant.PACK_ALIGN} == 0 を"
            "満たしていない、またはモジュール構成が変わった）"
        )
    return handles, attached


def latent_stage(
    name: str,
    recipe: Recipe,
    args: argparse.Namespace,
    base_frames: Mapping[str, int] | None,
) -> LatentResult:
    """テキスト → latent（構成ぶんの丸めを当てて full-loop を回す）。"""
    from tokenizers import Tokenizer

    source, modules, config, text_config, model_config = load_modules(
        args.model_dir, args.source_dir
    )
    # MUST: 丸めは load の直後・golden の採取より前（`irodori.export.fake_quant` の順序 MUST）。
    # 対象が空になるのは `i8-codec-only`（latent 側の役割を 1 つも含まない）だけで、そのときは
    # 丸めそのものを呼ばない — `fake_quant` の「0 本は fail loudly」は正しい規律なので、
    # ここで例外にせず**呼ばない**ことで満たす。
    #
    # 校正付き構成だけは順序が違う（丸めがグラフより後）— 校正入力は「素の f32 で回した参照
    # ループ」から採るので、グラフを組んでその上で pristine の full-loop を 1 周してからで
    # ないと丸められない。**丸めより前**という MUST は保たれている（このブロックまでは
    # どの重みも 1 ビットも動いていない）。
    scoped = role_modules(modules, recipe.roles)
    reports = (
        apply_weight_quant(recipe, scoped, args.kmeans_fit_stride)
        if recipe.weight is not None and recipe.calib is None and scoped
        else {}
    )
    patch.apply_patches()
    # 活性シムは重みの丸めの**後**（重みは実行前に決まり、活性は実行時に決まる — 両者は独立
    # だが、報告の順序を「重み → 活性」で揃えると素通りの切り分けが 1 本の出力で済む）。
    handles, act_quant_linears = attach_dit_act_quant(recipe, modules)
    if act_quant_linears:
        print(f"[{name}] 活性量子化シム: nn.Linear {act_quant_linears} 本（dit）", flush=True)
    graphs = build_graphs(modules, config, model_config)
    # サイズ試算の計数は**配布グラフ**から採る（{@link graph_scans}）— ラッパは丸め済みの
    # モジュールを抱えるだけなので、ここで数えても値には触れない。
    scans = (
        graph_scans(role_graphs(graphs), recipe.roles, recipe.op_types)
        if recipe.method is not None
        else {}
    )
    speaker_max = ex.speaker_sym_max(model_config)
    caps = {
        "text": int(model_config["max_text_len"]),
        "speaker": speaker_max + 1,
        "caption": int(model_config["max_caption_len"]),
    }
    tokenizer = Tokenizer.from_file(str(args.model_dir / ex.TOKENIZER_FILE))

    # グラフ一式を**既定引数で束ねる**のは、下の `del` が同名の局所を外すため（素の
    # クロージャだと、消えた後に呼ばれたとき NameError になる形が残る）。
    def run_case(
        case: ip.PipelineCase,
        graphs: ip.HostGraphs = graphs,
        source: Any = source,
        modules: Mapping[str, nn.Module] = modules,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return ip.run_case(
            case,
            graphs,
            source,
            modules[ex.TARGET_DURATION],
            tokenizer,
            text_config,
            model_config,
            config,
            caps,
            frames_override=None if base_frames is None else base_frames[case.name],
        )

    calib: tuple[CalibRig, CalibReport] | None = None
    if recipe.calib is not None:
        rig = build_calib_rig(graphs, run_case, args.calib_steps)
        reports, calib_report = apply_calib(recipe.calib, rig)
        scans = {ex.TARGET_DIT: RoleScan(rig.counts, ())}
        calib = (rig, calib_report)

    metas: dict[str, dict[str, Any]] = {}
    latents: dict[str, torch.Tensor] = {}
    for case in ip.PIPELINE_CASES:
        started = time.perf_counter()
        tensors, meta = run_case(case)
        meta["elapsed"] = round(time.perf_counter() - started, 1)
        metas[case.name] = meta
        latents[case.name] = tensors["z"].detach().clone()
        print(
            f"[{name}/{case.name}] S={meta['S']} predicted={meta.get('predictedS', meta['S'])}"
            f" forwards={meta['forwards']} {meta['elapsed']:.0f}s",
            flush=True,
        )
    act_quant.detach_act_quant(handles)
    del graphs, modules, source
    gc.collect()
    return LatentResult(metas, latents, reports, act_quant_linears, scans, calib)


def w4_payload(recipe: Recipe, scans: Mapping[str, RoleScan], fit_stride: int) -> dict[str, Any]:
    """w4 構成の対象規模・除外一覧・サイズ試算（JSON へそのまま載る形）。

    `tables` は表を張る成果物の数 = 対象を持つ役割数（`kmeans:shared` の表は役割ごとに
    1 枚 — {@link W4_KMEANS_SHARED}）。他の 3 方式では式に効かない。**校正付き構成では
    表の射程が層内**なので、代わりに対象の層数を渡す（{@link _calib_codebook_bits}）。
    """
    spec: W4Method | CalibConfig | None = recipe.method or recipe.calib
    if spec is None:
        raise SystemExit(f"w4 構成でないレシピ（weight={recipe.weight}）の試算を求められた")
    counts = {role: scan.counts for role, scan in scans.items()}
    total = total_counts(list(counts.values()))
    tables = (
        total.modules
        if recipe.calib is not None
        else sum(1 for item in counts.values() if item.modules)
    )
    size = project_size(spec, total, tables)
    return {
        "method": spec.name,
        "groupSize": W4_GROUP_SIZE,
        "opTypes": [cls.__name__ for cls in recipe.op_types],
        # MUST: 部分標本の表と全量の表は別物になりうるので、使った事実を数値の横へ出す
        # （`karume.quant_methods.fake_quant_kmeans` の MUST）。
        "fitStride": fit_stride,
        "roles": {
            role: {"modules": item.modules, "channels": item.channels, "elements": item.elements}
            for role, item in sorted(counts.items())
        },
        "excluded": [
            {
                "role": role,
                "fqn": item.fqn,
                "axisLength": item.axis_length,
                "elements": item.elements,
            }
            for role, scan in sorted(scans.items())
            for item in scan.excluded
        ],
        "size": {
            "formula": size.formula,
            "tables": size.tables,
            "modules": size.counts.modules,
            "channels": size.counts.channels,
            "elements": size.counts.elements,
            "bits": size.bits,
            "bitsPerWeight": size.bits_per_weight,
            "projectedMiB": size.projected_mib,
            "f32MiB": size.f32_mib,
        },
    }


def run_config(name: str, args: argparse.Namespace) -> dict[str, Any]:
    """1 構成ぶんを走らせ、per-config の JSON を書いて返す。"""
    started = time.perf_counter()
    recipe = RECIPES[name]
    check_fit_stride(recipe, args.kmeans_fit_stride)
    base_frames = None
    goldens = None
    if name == BASE_CONFIG:
        # MUST: 突合先の有無は**走らせる前**に確かめる（full-loop を回し切った後に
        # 「golden が無い」で落ちると、数十分ぶんの計算が捨てになる）。
        goldens = {case.name: golden_z(args.golden_dir, case.name) for case in ip.PIPELINE_CASES}
    else:
        base_json = args.out / f"{BASE_CONFIG}.json"
        if not base_json.is_file():
            raise SystemExit(
                f"基準構成の成果物が無い: {base_json}"
                f"（先に `--config {BASE_CONFIG}` を回す — 比較の基準は f32）"
            )
        base_meta = json.loads(base_json.read_text(encoding="utf-8"))
        base_frames = {case: entry["S"] for case, entry in base_meta["cases"].items()}
    if recipe.act_quant:
        # MUST: ここも**走らせる前**に確かめる（活性シムの素通り検出は重みだけの構成との差で
        # しか出せないので、無ければこの構成を回しても答えが出ない）。
        for case in ip.PIPELINE_CASES:
            weight_only_path(args.out, case.name)

    metas, latents, reports, act_quant_linears, scans, calib = latent_stage(
        name, recipe, args, base_frames
    )

    stage = build_decoder(
        args.codec_source_dir, args.codec_model_dir, recipe, args.kmeans_fit_stride
    )
    decoder, sample_rate, codec_report = stage.graph, stage.sample_rate, stage.report
    if stage.scan is not None:
        scans[ROLE_CODEC] = stage.scan
    audios: dict[str, torch.Tensor] = {}
    for case, latent in latents.items():
        with torch.no_grad():
            audios[case] = decoder(latent).reshape(-1).detach().clone()
        print(f"[{name}/{case}] decoded {audios[case].shape[0]} samples", flush=True)
    del decoder, stage
    gc.collect()

    args.out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    for case, meta in metas.items():
        audio, latent = audios[case], latents[case]
        path = args.out / f"{name}.{case}.wav"
        write_wav(path, audio, sample_rate)
        save_file(
            {"z": latent.contiguous(), "audio": audio.contiguous()},
            str(args.out / f"{name}.{case}{CASE_SUFFIX}"),
        )
        entry: dict[str, Any] = {
            "S": meta["S"],
            "predictedS": meta.get("predictedS", meta["S"]),
            "forwards": meta["forwards"],
            "predictedFrames": meta["duration"]["predictedFrames"],
            "predictedSeconds": round(meta.get("predictedS", meta["S"]) / ex.CODEC_FRAME_RATE, 3),
            "audioSeconds": round(int(audio.shape[0]) / sample_rate, 3),
            "samples": int(audio.shape[0]),
            "zAbsMax": meta["zAbsMax"],
            "cfgEffectMaxAbs": meta["cfgEffectMaxAbs"],
            "elapsed": meta["elapsed"],
            "wav": str(path),
            "wavSha256": sha256(path.read_bytes()),
        }
        if name != BASE_CONFIG:
            base = base_payload(args.out, case)
            base_z, base_audio = base["z"].to(torch.float32), base["audio"].to(torch.float32)
            if latent.shape != base_z.shape or audio.shape != base_audio.shape:
                raise SystemExit(
                    f"{case}: 基準と形が違う（z {tuple(latent.shape)} vs"
                    f" {tuple(base_z.shape)} / 波形 {tuple(audio.shape)} vs"
                    f" {tuple(base_audio.shape)}）— 基準を焼き直したなら全構成を回し直す"
                )
            entry["vsBase"] = {
                "zBitEqual": bool(torch.equal(latent, base_z)),
                "zRelRms": rel_rms(latent, base_z),
                "zMaxAbsDiff": float((latent - base_z).abs().max()),
                "wavBitEqual": bool(torch.equal(audio, base_audio)),
                "wavSnrDb": snr_db(audio, base_audio),
                "wavLsdDb": log_spectral_distance(audio, base_audio),
                "wavRelRms": rel_rms(audio, base_audio),
                "wavMaxAbsDiff": float((audio - base_audio).abs().max()),
            }
        if recipe.act_quant:
            # 素通り検出の観測（評定は {@link run_gates} が持つ）— **重みが同じ構成との差**を
            # 見るので、ここが 0 なら活性シムが 1 本も効いていない。
            weight_only = weight_only_payload(args.out, case)
            other_z = weight_only["z"].to(torch.float32)
            other_audio = weight_only["audio"].to(torch.float32)
            if latent.shape != other_z.shape or audio.shape != other_audio.shape:
                raise SystemExit(
                    f"{case}: {WEIGHT_ONLY_BASE} と形が違う（z {tuple(latent.shape)} vs"
                    f" {tuple(other_z.shape)} / 波形 {tuple(audio.shape)} vs"
                    f" {tuple(other_audio.shape)}）— 時間グリッドの固定が効いていない"
                )
            entry["vsWeightOnly"] = {
                "config": WEIGHT_ONLY_BASE,
                "zBitEqual": bool(torch.equal(latent, other_z)),
                "zRelRms": rel_rms(latent, other_z),
                "zMaxAbsDiff": float((latent - other_z).abs().max()),
                "wavSnrDb": snr_db(audio, other_audio),
                "wavLsdDb": log_spectral_distance(audio, other_audio),
            }
        entries[case] = entry

    payload = {
        "config": name,
        "weight": recipe.weight,
        "roles": list(recipe.roles),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "sampleRate": sample_rate,
        "frameRate": ex.CODEC_FRAME_RATE,
        "fakeQuant": reports,
        "codecFakeQuant": codec_report,
        "actQuantLinears": act_quant_linears,
        "cases": entries,
        "gates": run_gates(name, recipe, latents, entries, goldens),
        "elapsed": round(time.perf_counter() - started, 1),
    }
    if recipe.method is not None or recipe.calib is not None:
        payload["w4"] = w4_payload(recipe, scans, args.kmeans_fit_stride)
    if calib is not None:
        rig, calib_report = calib
        payload["calib"] = {
            "method": f"{calib_report.method}/{calib_report.grid}",
            "target": CALIB_TARGET,
            "stages": calib_report.stages,
            "roundedModules": calib_report.modules,
            "scanModules": rig.counts.modules,
            "steps": dict(rig.steps),
            "batches": len(rig.batches),
            "tokens": rig.tokens,
            "quantReport": calib_report.describe(),
        }
    (args.out / f"{name}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


# ---- 検証ゲート -------------------------------------------------------------


def run_gates(
    name: str,
    recipe: Recipe,
    latents: Mapping[str, torch.Tensor],
    entries: Mapping[str, Mapping[str, Any]],
    goldens: Mapping[str, torch.Tensor] | None,
) -> dict[str, Any]:
    """恒真化と scope 漏れの検出器（赤は最後にまとめて上げる）。

    ① `f32` の z が full-loop golden と**バイト一致**（この台本の経路が本物と同じ計算である
       ことの実証 — 外れたら測定値は全部意味を失う）
    ② 各構成の z / 波形が**変わるべきところだけ変わる**（モジュール docstring の直交性の門）。
       「変わること」だけを門にすると丸めの素通りしか捕まらず、「変わらないこと」だけを
       門にすると scope 漏れしか捕まらない — 両方を期待値表から出す
    ③ 活性シムを持つ構成は**重みだけの構成とも z が違う**（素通り検出）。基準との差だけでは
       重みの丸めで説明が付いてしまい、活性シムが 1 本も効いていなくても緑になる
    """
    gates: dict[str, Any] = {}
    failures: list[str] = []
    if goldens is not None:
        matched = {
            case: bool(torch.equal(latent, goldens[case])) for case, latent in latents.items()
        }
        gates["z_bit_exact_vs_golden"] = {"cases": matched}
        failures += [
            f"{case}: f32 の z が full-loop golden とバイト一致しない（経路の恒真化検査）"
            for case, ok in matched.items()
            if not ok
        ]
    else:
        observed = {
            case: {
                "zChanged": not entry["vsBase"]["zBitEqual"],
                "wavChanged": not entry["vsBase"]["wavBitEqual"],
            }
            for case, entry in entries.items()
        }
        expectations: dict[str, dict[str, bool]] = {}
        for case in observed:
            # 期待は**ケース条件**で割る — 参照なしのケースでは speaker 経路が走らない
            # （{@link REFERENCE_ONLY_ROLES}）ので、その分の役割を期待から外す。
            effective = set(recipe.roles) & LATENT_ROLES
            if not CASE_HAS_REFERENCE[case]:
                effective -= REFERENCE_ONLY_ROLES
            expect_z = bool(effective)
            expectations[case] = {
                "zChanged": expect_z,
                "wavChanged": expect_z or ROLE_CODEC in recipe.roles,
            }
        gates["orthogonality"] = {"expected": expectations, "observed": observed}
        for case, seen in observed.items():
            expect_z = expectations[case]["zChanged"]
            expect_wav = expectations[case]["wavChanged"]
            if seen["zChanged"] != expect_z:
                failures.append(
                    f"{case}: z が「{'変わる' if expect_z else '変わらない'}」期待に反した"
                    f"（役割 {list(recipe.roles)} — 丸めの素通りか scope 漏れ）"
                )
            if seen["wavChanged"] != expect_wav:
                failures.append(
                    f"{case}: 波形が「{'変わる' if expect_wav else '変わらない'}」期待に反した"
                    f"（役割 {list(recipe.roles)}）"
                )
        if recipe.act_quant:
            moved = {
                case: not entry["vsWeightOnly"]["zBitEqual"] for case, entry in entries.items()
            }
            gates["act_quant_changes_the_latent"] = {"versus": WEIGHT_ONLY_BASE, "cases": moved}
            failures += [
                f"{case}: z が {WEIGHT_ONLY_BASE} とビット一致した（活性シムが素通りしている"
                " — 重みだけの構成と同じ数を w8a8 の名前で測っている）"
                for case, ok in moved.items()
                if not ok
            ]
    if recipe.weight is not None:
        gates["quantized_roles"] = list(recipe.roles)
    gates["failures"] = failures
    return gates


# ---- レポート ---------------------------------------------------------------


def collect(out_dir: Path) -> dict[str, dict[str, Any]]:
    """`--out` に残っている per-config の JSON を構成順に集める。

    MUST: 直交性の門は**ここで評定し直す**（保存 JSON の `gates` を再掲しない）— 門の期待は
    コードが正本で、測定後に期待側の誤りを直したとき、保存済みの評定が古い期待のまま
    レポートへ蘇るため（実例: speaker のケース条件 — no-ref はゼロ短絡で不変が正しい）。
    基準 `f32` の golden ビット一致だけはテンソル比較（測定時にしかできない）なので保存値を
    使う。観測（`vsBase` / `vsWeightOnly` のビット等値）は測定の事実で、評定と違い保存してよい側。
    """
    found: dict[str, dict[str, Any]] = {}
    for name in RECIPES:
        path = out_dir / f"{name}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if name != BASE_CONFIG:
                payload["gates"] = run_gates(name, RECIPES[name], {}, payload["cases"], None)
            found[name] = payload
    return found


def drift_table(collected: Mapping[str, Mapping[str, Any]]) -> str:
    """**最上段の表** — S が動いたか（混成表の裁定はこの 1 枚で決まる）。"""
    header = ["config", "case", "predicted S", "基準 S", "S ドリフト", "forwards", "発話長 (s)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    base = collected.get(BASE_CONFIG, {}).get("cases", {})
    for name, payload in collected.items():
        for case, entry in payload["cases"].items():
            reference = base.get(case, {}).get("predictedS")
            drift = "—" if reference is None else f"{entry['predictedS'] - reference:+d}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        case,
                        str(entry["predictedS"]),
                        "—" if reference is None else str(reference),
                        "一致" if drift == "+0" else drift,
                        str(entry["forwards"]),
                        f"{entry['predictedSeconds']:.2f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def quality_table(collected: Mapping[str, Mapping[str, Any]]) -> str:
    """品質（**基準と同じ時間グリッド上**での比較）。"""
    header = ["config", "case", "latent relRMS", "latent max|Δ|", "SNR (dB)", "LSD (dB)", "秒"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for name, payload in collected.items():
        for case, entry in payload["cases"].items():
            versus = entry.get("vsBase")
            if versus is None:
                row = ["0（基準）", "0", "inf（基準）", "0（基準）"]
            else:
                snr = versus["wavSnrDb"]
                row = [
                    f"{versus['zRelRms']:.4e}",
                    f"{versus['zMaxAbsDiff']:.4e}",
                    "inf" if math.isinf(snr) else f"{snr:.2f}",
                    f"{versus['wavLsdDb']:.2f}",
                ]
            lines.append("| " + " | ".join([name, case, *row, f"{entry['elapsed']:.0f}"]) + " |")
    return "\n".join(lines)


def size_table(collected: Mapping[str, Mapping[str, Any]]) -> str:
    """w4 構成のサイズ試算（**式による投影**・実測ではない）。

    品質だけ並べると表のコストが見えない（`kmeans:shared` は成果物ごとに 16×f32 の表が要る）
    ので、同じレポートに式ごと載せる。
    """
    header = ["config", "方式", "本数", "要素", "式", "bpw", "投影 MiB", "f32 MiB", "縮小率"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for name, payload in collected.items():
        size = payload.get("w4", {}).get("size")
        if size is None:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    payload["w4"]["method"],
                    str(size["modules"]),
                    f"{size['elements']:,}",
                    size["formula"],
                    f"{size['bitsPerWeight']:.3f}",
                    f"{size['projectedMiB']:.1f}",
                    f"{size['f32MiB']:.1f}",
                    f"{size['projectedMiB'] / size['f32MiB']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def excluded_lines(collected: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """g32 非整列で対象から外した重みの一覧（構成ごと・**必ず出力へ載せる**）。

    黙って外すと「全役割を丸めた」と読める表の裏で、割り切れない層だけ f32 のまま残る。
    """
    lines: list[str] = []
    for name, payload in collected.items():
        excluded = payload.get("w4", {}).get("excluded")
        if excluded is None:
            continue
        if not excluded:
            lines.append(f"- {name}: g32 非整列による除外なし")
            continue
        lines.append(f"- {name}: g32 非整列 {len(excluded)} 本を除外")
        lines += [
            f"    - {item['role']} / {item['fqn']} 軸長 {item['axisLength']}"
            f"（{item['elements']:,} 要素）"
            for item in excluded
        ]
    return lines


def build_report(collected: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/irodori/measure_quant.py",
        "torch": torch.__version__,
        "base": BASE_CONFIG,
        "method": {
            "time_grid": "基準以外は f32 の S を外から固定（波形長を揃えて SNR を定義可能に"
            "する）。構成自身の予測は predictedS",
            "roles": list(ROLES),
            "latent_roles": sorted(LATENT_ROLES),
            "metrics": "SNR は波形の残差（位相ずれに弱い）/ LSD は対数振幅スペクトルの距離"
            f"（n_fft={STFT_N_FFT} hop={STFT_HOP}・床 −100dB・位相ずれに鈍い）。両方で読む",
            "trim": "末尾トリムと秒切り出しは通さない（構成間で長さを揃えるため）",
            "act_quant": f"w8a8 は DiT の nn.Linear 入力へ per-token i8 の fake-quant を掛ける"
            f"（karume.act_quant — ランタイム quantize-rows.ts の鏡像）。素通り検出は"
            f" {WEIGHT_ONLY_BASE} との z ビット等値",
            "w4": f"方式は g={W4_GROUP_SIZE} 固定で比較（ADR 0069 追記 5 の 3）。対象 op は"
            "構成ごと（正本は各 config JSON の w4.opTypes — 全役割形 = i8 と同じ 5 種 /"
            " i4-linear = Linear のみ）。i4 の実行経路は linear 限定（決定 5）なので全役割形は"
            "出荷できる形ではなく品質の上限を測る側、i4-linear が今日の配布対応形。サイズ列は"
            "**式による投影**で、計数は配布グラフに載る重みだけ",
            "calib": f"校正付き構成（{' / '.join(CALIB_NAMES)}）は core の `karume.quant_calib`"
            f" で DiT block 列の linear（{CALIB_TARGET}）だけを stage 逐次に丸める。校正入力は"
            "参照ループ（f32・CFG 込み）の step 横断で、step ごとに先頭 block への入力を"
            "捕まえたもの（使った step 数は各 config JSON の calib.steps）。**対象集合は"
            " i4-linear と違う**ので、bpw は同じでも品質の行は直接は比較できない",
            "verdict": "最終裁定は聴感（ユーザー）— WAV は同一テキスト・同一 seed で並ぶ",
        },
        "configs": dict(collected),
        "failures": {
            name: payload["gates"]["failures"]
            for name, payload in collected.items()
            if payload["gates"]["failures"]
        },
    }


# ---- 対象規模の下見（`--scan`）----------------------------------------------

#: 下見で並べる対象 op 集合。`linear` は i4 の実行経路がそのまま受けられる形（ADR 0069
#: 決定 5）、`all` は測定用に広げた {@link W4_OP_TYPES}。**linear 限定形は品質を測らない**
#: — 数えれば conv 主体かどうかが分かり、conv 主体なら linear だけ丸めても配布サイズが
#: 動かないので、full-loop を数十分回す価値が無い（回すかどうかはこの表で決める）。
SCAN_OP_SETS: Mapping[str, tuple[type[nn.Module], ...]] = MappingProxyType(
    {"linear": (nn.Linear,), "all": W4_OP_TYPES}
)

SCAN_FILE = "w4-scan.json"

#: 下見で試算を並べる方式（構成表の 4 種と同じ順・同じ実体）。
SCAN_METHODS: tuple[W4Method, ...] = (W4_RTN, W4_NF4, W4_MXFP4, W4_KMEANS_SHARED)


def scan_graphs(args: argparse.Namespace) -> dict[str, nn.Module]:
    """配布グラフ 7 本（6 役割 + コーデックの decode 経路）を**丸めずに**組む。"""
    source, modules, config, _text_config, model_config = load_modules(
        args.model_dir, args.source_dir
    )
    patch.apply_patches()
    graphs = role_graphs(build_graphs(modules, config, model_config))
    stage = build_decoder(args.codec_source_dir, args.codec_model_dir, RECIPES[BASE_CONFIG])
    del source
    return {**graphs, ROLE_CODEC: stage.graph}


def scan_report(args: argparse.Namespace) -> dict[str, Any]:
    """役割 × op 集合の対象規模とサイズ試算（full-loop を 1 回も回さない）。"""
    graphs = scan_graphs(args)
    op_sets: dict[str, Any] = {}
    for label, op_types in SCAN_OP_SETS.items():
        scans = {role: scan_targets(graph, op_types) for role, graph in graphs.items()}
        counts = {role: scan.counts for role, scan in scans.items()}
        tables = sum(1 for item in counts.values() if item.modules)
        total = total_counts(list(counts.values()))
        op_sets[label] = {
            "opTypes": [cls.__name__ for cls in op_types],
            "roles": {
                role: {
                    "modules": item.modules,
                    "channels": item.channels,
                    "elements": item.elements,
                    "f32MiB": item.elements * 4 / 1024**2,
                }
                for role, item in counts.items()
            },
            "excluded": [
                {
                    "role": role,
                    "fqn": item.fqn,
                    "axisLength": item.axis_length,
                    "elements": item.elements,
                }
                for role, scan in scans.items()
                for item in scan.excluded
            ],
            "total": {
                "modules": total.modules,
                "channels": total.channels,
                "elements": total.elements,
                "f32MiB": total.elements * 4 / 1024**2,
            },
            "methods": {},
        }
        for method in SCAN_METHODS:
            size = project_size(method, total, tables)
            op_sets[label]["methods"][method.name] = {
                "formula": size.formula,
                "tables": size.tables,
                "bits": size.bits,
                "bitsPerWeight": size.bits_per_weight,
                "projectedMiB": size.projected_mib,
                "f32MiB": size.f32_mib,
                "ratio": size.projected_mib / size.f32_mib,
            }
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/irodori/measure_quant.py --scan",
        "groupSize": W4_GROUP_SIZE,
        "note": "計数は配布グラフに載る重みだけ（丸める木は役割モジュール = i8 と同じ集合）。"
        "サイズは式による投影で、実測ではない",
        "opSets": op_sets,
    }


def scan_scale_table(payload: Mapping[str, Any]) -> str:
    """役割 × op 集合の対象規模（**本数・チャネル・要素**）。"""
    header = ["op 集合", "役割", "本数", "チャネル", "要素", "f32 MiB"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for label, entry in payload["opSets"].items():
        rows = [*entry["roles"].items(), ("合計", entry["total"])]
        lines += [
            "| "
            + " | ".join(
                [
                    label,
                    role,
                    str(item["modules"]),
                    f"{item['channels']:,}",
                    f"{item['elements']:,}",
                    f"{item['f32MiB']:.2f}",
                ]
            )
            + " |"
            for role, item in rows
        ]
    return "\n".join(lines)


def scan_size_table(payload: Mapping[str, Any]) -> str:
    """op 集合 × 方式のサイズ試算（**式による投影**・実測ではない）。"""
    header = ["op 集合", "方式", "式", "bpw", "投影 MiB", "f32 MiB", "縮小率"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for label, entry in payload["opSets"].items():
        lines += [
            "| "
            + " | ".join(
                [
                    label,
                    name,
                    size["formula"],
                    f"{size['bitsPerWeight']:.3f}",
                    f"{size['projectedMiB']:.1f}",
                    f"{size['f32MiB']:.1f}",
                    f"{size['ratio']:.3f}",
                ]
            )
            + " |"
            for name, size in entry["methods"].items()
        ]
    return "\n".join(lines)


def scan_excluded_lines(payload: Mapping[str, Any]) -> list[str]:
    """g32 非整列で対象から外れる重みの一覧（op 集合ごと）。"""
    lines: list[str] = []
    for label, entry in payload["opSets"].items():
        if not entry["excluded"]:
            lines.append(f"- {label}: g32 非整列による除外なし")
            continue
        lines.append(f"- {label}: g32 非整列 {len(entry['excluded'])} 本")
        lines += [
            f"    - {item['role']} / {item['fqn']} 軸長 {item['axisLength']}"
            f"（{item['elements']:,} 要素）"
            for item in entry["excluded"]
        ]
    return lines


def run_scan(args: argparse.Namespace) -> None:
    """{@link scan_report} を走らせて JSON を書き、markdown 表を stdout へ出す。"""
    payload = scan_report(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / SCAN_FILE).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print("w4 対象規模（配布グラフに載る重みだけ・丸めていない）:")
    print(scan_scale_table(payload))
    print()
    print("g32 非整列の除外:")
    print("\n".join(scan_excluded_lines(payload)))
    print()
    print("サイズ試算（式による投影・実測ではない）:")
    print(scan_size_table(payload))
    print()
    print(f"scan → {args.out / SCAN_FILE}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=ex.DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=ex.DEFAULT_SOURCE_DIR)
    parser.add_argument("--codec-model-dir", type=Path, default=dv.DEFAULT_MODEL_DIR)
    parser.add_argument("--codec-source-dir", type=Path, default=dv.DEFAULT_SOURCE_DIR)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--config",
        choices=tuple(RECIPES),
        default=None,
        help="走らせる構成（1 実行 = 1 構成）。省略すると実測はせず、"
        f"--out に残っている成果物から {REPORT_FILE} と表を組み直すだけ",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="w4 の対象規模とサイズ試算だけを出して終わる（full-loop は回さない）",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=ip.NUM_STEPS,
        help=f"校正付き構成が 1 ケースあたり使う step 数（既定 = 参照ループ全 {ip.NUM_STEPS}"
        " step）。捕まえ切った時点で参照ループを畳むので、縮小 smoke ではここを下げる。"
        "使った値は出力へ載る",
    )
    parser.add_argument(
        "--kmeans-fit-stride",
        type=int,
        default=1,
        help="kmeans:shared の**表の fit だけ**を等間隔部分標本にする（適用は常に全量）。"
        "dit 役割の全量 fit は f64 の作業領域まで含めて 20GB 超になるので、"
        "kmeans-shared-all は実メモリに合わせてここを上げる。使った値は出力へ載る",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.calib_steps <= ip.NUM_STEPS:
        parser.error(f"--calib-steps は 1〜{ip.NUM_STEPS}（参照ループの step 数）")

    if args.scan:
        run_scan(args)
        return

    if args.config is not None:
        payload = run_config(args.config, args)
        print(f"[{args.config}] {payload['elapsed']:.0f}s → {args.out / f'{args.config}.json'}")

    collected = collect(args.out)
    if not collected:
        raise SystemExit(f"{args.out} に構成の成果物が 1 つも無い（--config で走らせる）")
    report = build_report(collected)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print("S ドリフト（混成表の裁定はこの表）:")
    print(drift_table(collected))
    print()
    print("品質（基準と同じ時間グリッド上）:")
    print(quality_table(collected))
    exclusions = excluded_lines(collected)
    if exclusions:
        print()
        print("w4 サイズ試算（式による投影・実測ではない）:")
        print(size_table(collected))
        print()
        print("g32 非整列の除外:")
        print("\n".join(exclusions))
    print()
    print(f"report → {args.out / REPORT_FILE}")
    if report["failures"]:
        raise AssertionError(
            "検証ゲートが赤: "
            + " / ".join(
                f"{name}: {'・'.join(items)}" for name, items in report["failures"].items()
            )
        )
    print("gates: all green")


if __name__ == "__main__":
    main()
