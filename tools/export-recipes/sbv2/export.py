"""SBV2 音響チェーンを IR v1 + golden io へ書き出す台本（ADR 0013 の emit ターゲット）。

ターゲットは 5 本（ADR 0013 の全量）:

- **`dp`**（M1-P3 波 1）— DurationPredictor 単体。パッチ層も語彙拡張も要らず、「実重み →
  export → 実 GPU E2E」の貫通を先に取るための足場。
- **`front`**（波 6）— enc_p + dp + sdp(reverse) の融合 1 グラフ。動的 `P`（音素数）で、
  `sbv2.patch` のパッチ層（spline 分岐フリー化 / 相対位置注意の gather 化 /
  FFN の pad 畳み込み）を当てて初めて export できる。
- **`flow`**（波 7）— TransformerCouplingBlock の reverse。動的 `T`（フレーム数）で、相対
  位置注意の `(T,T)` 表は**グラフ入力**（焼き込みは 134MB — ADR 0013）。
- **`dec`**（波 7）— HiFi-GAN Generator。パッチ不要で、前処理は `remove_weight_norm` のみ。
- **`voice`**（波 7）— flow + dec の融合 1 グラフ。**この E2E が緑になった時点で SBV2 の
  全チェーンが成立する**。

    uv sync --all-groups                                    # tools/export-recipes/ で 1 回
    uv run --group sbv2 python -m sbv2.export               # 全ターゲットを emit
    uv run --group sbv2 python -m sbv2.export --target front
    uv run --group sbv2 python -m sbv2.export --dtype f16   # → outputs/series/sbv2-FN4-f16/
    uv run --group sbv2 python -m sbv2.export --dtype i8    # → outputs/series/sbv2-FN4-i8/
    uv run --group sbv2 python -m sbv2.export --dtype i4 --target front --target voice
    uv run --group sbv2 python -m sbv2.export --verify flow  # 参照実装との eager 同値検証

NOTE: sync が `--all-groups` なのは workspace の venv が 1 つだからで（`tools/pyproject.toml`）、
`--group sbv2` だけを sync すると他 family のグループが**外れる**（実測）。

MUST: `--verify` と emit は**同一プロセスで併用できない**（CLI が機械的に拒否する）。
パッチはクラス属性のプロセス全域差し替えなので、emit 側が先にパッチを当てると「パッチ前の
参照」が採れなくなり、同値検証が恒真化して偽 PASS する（ADR 0013）。検証自体も
「全ケースの参照値を確定 → 変更 → 比較」の順序を守り、順序が破れていれば参照採取の直前で
落とす — front / flow / voice は `sbv2.patch.patches_applied()`、**dec は逆向き**に
「weight_norm 由来のパラメータがまだ残っていること」を見る（汚染源が remove だから）。

MUST: `--verify` は**ターゲットを 1 つだけ取る**（`--verify front` / `--verify dec` …）。
複数の検証を並べられる形にすると「MHA パッチ系どうしは排他 / dec の remove は voice と
だけ排他 / 丸めは全てと排他」という対ごとの排他表を CLI が持つことになり、表の穴が
そのまま偽 PASS になる。**1 プロセス 1 検証**なら汚染の組み合わせが構造的に存在しない。

重みは配布物に含まれない（`inputs/` は `.gitignore` 済み）。入手手順は README を参照。

出力レイアウト（Deno 側 `packages/runtime/tests/e2e_sbv2_test.ts` が列挙する）。系列名は
`--model-dir` のディレクトリ名から導く（既定の `inputs/sbv2/FN4/` なら `sbv2-FN4`）:

    outputs/series/sbv2-FN4/<target>/model.safetensors     重み・定数 + __metadata__.karume_ir
    outputs/series/sbv2-FN4/<target>/io.<case>.safetensors 入力と torch CPU での期待出力

io のテンソルキー規約は tiny golden / DeBERTa と同じ（`input.<グラフ入力名>` /
`output.<位置>`）。

## 格納 dtype の系列（ADR 0018 / 0019 / 0069）

`--dtype f16` / `--dtype i8` / `--dtype i4` はそれぞれ**別系列**（`sbv2-FN4-f16/` /
`sbv2-FN4-i8/` / `sbv2-FN4-i4/`）へ書く — f32 系列と同居させると既存 E2E の網（f32 の
tolerance）が黙って別の資産に掛かる。丸め（fake-quant）は共有の
`quantize.round_weights_to_f16` / `quantize.fake_quant_int8` / `quantize.fake_quant_int4` を
**remove_weight_norm / パッチ適用の後・参照と golden の採取の前**に当てる
（`_fake_quant` の順序 MUST）。

`--dtype i4` だけは**混成**（適格な `nn.Linear` / `nn.Conv1d` = i4 group32・残りは i8
per-channel）で、配布形では `w4` quant の `front` / `voice` 席に入る（`sbv2/distribution.py`）。
i4 の実行経路は linear / embedding / conv1d の重みスロット限定（ADR 0069 決定 5 とその追補）で、
`nn.ConvTranspose1d` と depthwise conv（`groups > 1`）と行長が group32 で割り切れない重みは
i8 側へ落ちるので、系列としては混成にしかならない。

MUST: `--dtype` は **emit 専用**（`--sym-max` と同じ扱い）。`--verify` は格納形式を見ない
eager 比較で、しかも dec / voice では丸めを remove の**後**にしか当てられないのに参照は
remove の**前**に採るため、併用すると「丸めた側 vs 丸めていない側」の比較になって
`bit_exact` の主張が壊れる。CLI が機械的に拒否する。
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from karume.convert import normalize_boundary_tensor
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QUANT_MODULE_TYPES,
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
    round_weights_to_f16,
)

from . import patch

#: 実重みの置き場。リポジトリ管理外（`.gitignore` の `inputs/`）で、手で配置する。
DEFAULT_MODEL_DIR = INPUTS_ROOT / "sbv2" / "FN4"

#: 書き出せる格納 dtype。`i8` は **2026-08-04 のユーザー裁定**で足した（ADR 0027 決定 1 の
#: 「i8 は足さない」の上書き — 記録は SBV2 w8 系列の ADR）。**w8a8 の受け皿ではない**点は
#: 変わらない（5 ターゲットとも conv1d が 86〜90% を占め linear は実質 0 GFLOP —
#: ADR 0025 決定⑤）。狙いは資産サイズとロード時間で、実行経路は ADR 0019 の w8a32
#: （i8 格納・計算は f32）そのもの。
#: `i4` は**混成**の系列名で、実体は「適格な `nn.Linear` / `nn.Conv1d` = i4 group32・それ以外
#: （非適格の linear / conv1d と `ConvTranspose1d` / embedding）= 従来どおり i8 per-channel」
#: （{@link BASE_WEIGHT_DTYPES} / {@link I4_MODULE_TYPES} / {@link _i4_module_names}）。
#: 適格外が必ず残る（`dec` の `ups` は転置レイアウト・`dp` の `convs_sep` は depthwise）ので、
#: 系列としては混成にしかなり得ない。net_g の linear は 6 本しかない（front 2 / voice 4）が、
#: **conv1d は 5 ターゲットとも本体**（ADR 0025 決定⑤）なので、conv1d の追補（波 J-5b）で
#: 初めて net_g 自身の配布サイズが動く。
#:
#: MUST: i4 の適格を 1 本も持たないターゲットに `i4` を渡すと `fake_quant_int4` が
#: 「対象 0 本」で fail loudly する（沈黙 i8 系列を作らないための門）。
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "i8", "i4")

#: 系列名 → `export_to_file` へ渡す**既定**の格納 dtype。i4 系列だけ既定が i8 で、適格な重みは
#: 1 本単位の `weight_dtype_overrides` で i4 へ振る（deberta の i4 混成系列と同形）。
BASE_WEIGHT_DTYPES: Mapping[str, str] = {"f32": "f32", "f16": "f16", "i8": "i8", "i4": "i8"}

#: i4 group32 で丸めるモジュール型（emit の `I4_WEIGHT_OPS` = linear / embedding / conv1d の
#: モジュール側の綴り）。**embedding は入れない** — net_g の語彙表（`enc_p.emb`）の i4 化は
#: 別の判断で、波 J-5b の射程外（`nn.ConvTranspose1d` は `nn.Conv1d` の派生ではないので、
#: `dec` の `ups` は型で自然に落ちる）。実際の適格はこの型に `groups == 1` と行長の整除を
#: 掛けた積（{@link _i4_module_names}）。
I4_MODULE_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv1d)


def default_out_root(model_dir: Path, dtype: str) -> Path:
    """生成物の既定の置き場（`outputs/series/sbv2-<実重みのディレクトリ名>{,-f16,-i8,-i4}/`）。

    ターゲット名（`dp` / `front`）のサブディレクトリは呼び出し側が 1 段掘る。

    話者名（`--model-dir` のディレクトリ名）を系列の綴りへ焼くのは、将来の多話者で系列
    どうしを衝突させないため — 綴りを `sbv2/` で共有すると、別の話者を書き出した瞬間に
    先の資産が黙って上書きされる。

    MUST: dtype ごとに別ディレクトリ（ADR 0018 / 0019）— 同居させると f32 系列の網（実測から
    導いたターゲット別 tolerance）が圧縮資産へ黙って掛かる。
    """
    suffix = "" if dtype == "f32" else f"-{dtype}"
    return SERIES_ROOT / f"sbv2-{model_dir.name}{suffix}"


TARGET_DP = "dp"
TARGET_FRONT = "front"
TARGET_FLOW = "flow"
TARGET_DEC = "dec"
TARGET_VOICE = "voice"
TARGETS = (TARGET_DP, TARGET_FRONT, TARGET_FLOW, TARGET_DEC, TARGET_VOICE)
CONFIG_FILE = "config.json"
STYLE_FILE = "style_vectors.npy"
MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: 記号次元 P（音素数）の上限 — **front 系（dp / front）の既定**。
SYM_MAX = 512

#: 記号次元 T（フレーム数）の上限 — **flow 系（flow / dec / voice）の既定**。
FLOW_SYM_MAX = 4096

#: ターゲット → 記号次元の上限。ADR 0013 の「sym_max はターゲット別既定」そのもので、
#: 取り違えは沈黙する（front を 4096 で出すと相対位置表の焼き込みが 134MB になり、flow を
#: 512 で出すと 512 フレーム = 約 6 秒より長い発話が Tmax 超過で落ちる）。**CLI の
#: `--sym-max` は既定 None** で、逸脱するときだけ明示させる（しかも単一ターゲット限定 —
#: 複数ターゲットに 1 つの値を配ると必ずどちらかが誤値になる）。
TARGET_SYM_MAX: dict[str, int] = {
    TARGET_DP: SYM_MAX,
    TARGET_FRONT: SYM_MAX,
    TARGET_FLOW: FLOW_SYM_MAX,
    TARGET_DEC: FLOW_SYM_MAX,
    TARGET_VOICE: FLOW_SYM_MAX,
}

#: 相対位置注意の窓幅。gather 化パッチはこの幅の埋め込み（2w+1 = 9 行）を前提に添字表を
#: 組み、エクスポータの定数畳み込みが Pmax で焼き込む。
EXPECTED_WINDOW_SIZE = 4

#: 話者埋め込みを引く話者 ID。実重みは 1 話者モデル（config の `n_speakers`）。
SPEAKER_ID = 0

#: 使うスタイルベクトルの行（`style_vectors.npy` は [スタイル数, 256]）。0 は平均スタイル。
STYLE_ID = 0

#: 乱数はここから派生させる（グローバル seed に依存しない — 再生成でバイト一致させる）。
SEED = 20260802

#: sdp reverse のノイズ倍率。**グラフには焼かない**（実行時ノブ — ADR 0013）: golden の
#: `z_noise` は「乗算済みの列」で、ランタイム側は倍率を知らないまま同じ数を再現する。
#: 参照実装 `StochasticDurationPredictor.forward(reverse=True)` の既定と同じ 0.8 を使う。
NOISE_SCALE = 0.8

#: golden ケース `(名前, 長さ, 末尾パディング長)`。**全ターゲットで共通**（Deno 側の
#: `EXPECTED_CASES` が 1 本の表でターゲット横断に等値検査する）。長さの意味はターゲットで
#: 変わる — front 系は P（音素数）、flow 系は T（フレーム数）。
#:
#: 長さは下限（2 — torch.export の 0/1 特殊化を避ける最小値）から front の宣言上限
#: SYM_MAX まで散らす。`padded` は **マスクの末尾に 0 を置く唯一のケース**で、マスク乗算が
#: 効いていることを踏む（外れるとパディング列の出力が 0 でなくなる）。
#:
#: NOTE: flow 系の宣言上限 4096 を踏む golden は**置かない**。相対位置表と注意スコアが
#: O(T²) なので T=4096 では表だけで io 1 ケース 134MB、dec の出力も 512·4096 = 210 万点に
#: なり、golden 資産としても実 GPU テストとしても割に合わない。上限近傍の挙動は
#: 「宣言上限に依存した実装（プランを Tmax で組む等）が無いこと」を長さの散らばりで踏む
#: 側に寄せる（P/T が 2 → 512 で 256 倍変わる 5 ケース）。
#: **grid-stride の縮退耐性はこの表では踏めない** — 必要 workgroup 数が 1 次元の dispatch 上限
#: （65535）を超えて初めて発動する経路で、T=512 では届かない（pad は T ≳ 2900 で超える）。
#: 担保は Deno 側の縮退ハーネス `packages/runtime/tests/gpu_gridstride_test.ts`
#: （dispatch 数を意図的に絞って全カーネル族に当てる）が持つ。
GOLDEN_CASES: tuple[tuple[str, int, int], ...] = (
    ("p2", 2, 0),
    ("p37", 37, 0),
    ("p203", 203, 0),
    ("p512", SYM_MAX, 0),
    ("padded", 16, 5),
)

#: `Sbv2Front.forward` の引数順（= IR の入力宣言順）。dynamic_shapes は位置で対応するので、
#: 例示入力の並べ方をここ 1 箇所に固定する（並べ替えると動的軸が別の入力に付く）。
FRONT_INPUT_ORDER = ("x", "x_mask", "tone", "language", "bert", "style_vec", "g", "z_noise")

#: front の出力名（IR の `output.<位置>` に対応。同値検証のレポート用）。
FRONT_OUTPUT_NAMES = ("logw_sdp", "logw_dp", "m_p", "logs_p")

#: パッチ同値検証のケース `(P, 末尾パディング長)`。golden より広く採る — 同値性は
#: golden の 5 点だけでなく **P の連続域**で成り立つべき性質で、特に `P ≤ window_size`
#: （相対位置埋め込みのスライス幅が P で変わる領域）と、その境界前後を踏む必要がある。
VERIFY_CASES: tuple[tuple[int, int], ...] = (
    (2, 0),
    (4, 0),
    (5, 0),
    (9, 0),
    (16, 5),
    (37, 0),
    (64, 0),
    (203, 0),
    (SYM_MAX, 0),
)

#: `FlowReverse` / `Sbv2Voice` / `dec` の入力順（= IR の入力宣言順）。front と同じく
#: dynamic_shapes は位置で対応するので、並べ方をここ 1 箇所に固定する。
FLOW_INPUT_ORDER = ("z_p", "y_mask", "g", "idx_k", "valid")
DEC_INPUT_ORDER = ("x", "g")

#: flow 系の同値検証ケース `(T, 末尾パディング長)`。golden より広く採る（同値性は golden の
#: 5 点ではなく T の連続域で成り立つべき性質）。**窓幅 4 の前後**（T ≤ w / T = w+1 / T > w）
#: を踏む — 相対位置表は `clamp(rel+w, 0, 2w)` なので小さい T では clamp が全域に効き、
#: 大きい T とは別の分岐を通る。
FLOW_VERIFY_CASES: tuple[tuple[int, int], ...] = (
    (2, 0),
    (4, 0),
    (5, 0),
    (9, 0),
    (16, 5),
    (24, 0),
    (50, 0),
    (137, 0),
    (203, 0),
    (SYM_MAX, 0),
)


class DurationPredictorGraph(nn.Module):
    """dp を「g 必須・引数名が IR の入力名」の形に固定する export 用ラッパ。

    素の `DurationPredictor.forward` は `g` が `Optional` で `g is not None` の分岐を
    持つ。ラッパで必須にして分岐を消し、入力名を recon の呼び名（h / x_mask / g）に
    揃える — IR の入力名は forward の引数名がそのまま出るので、ここが golden の
    `input.<name>` キーの正本になる。
    """

    def __init__(self, dp: nn.Module) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, h: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.dp(h, x_mask, g=g)


def load_net_g(model_dir: Path) -> tuple[nn.Module, Any]:
    """`inputs/sbv2/<名前>/` に配置された実重みから net_g を組み立て、eval で返す。

    ckpt は `*.safetensors` の一意存在を要求する（複数あると「どれを読んだか」が
    黙って変わる）。
    """
    from style_bert_vits2.models.hyper_parameters import HyperParameters
    from style_bert_vits2.models.infer import get_net_g

    # 非再帰の glob。系列は `outputs/series/` 側なので、ここには入力素材しか並ばない。
    ckpts = sorted(model_dir.glob("*.safetensors"))
    if len(ckpts) != 1:
        raise FileNotFoundError(
            f"{model_dir} の ckpt が一意でない（{len(ckpts)} 件: {[p.name for p in ckpts]}）"
        )
    config = model_dir / CONFIG_FILE
    if not config.is_file():
        raise FileNotFoundError(f"{config} が無い（ckpt と同じディレクトリに置く）")

    hps = HyperParameters.load_from_json(str(config))
    net_g = get_net_g(str(ckpts[0]), hps.version, "cpu", hps)
    net_g.eval()
    _assert_window_size(net_g)
    # weight_norm が 0 件と実測されている 4 モジュール（recon §3 — 残る 95 件は dec に集中）。
    # dp だけを見ていると、enc_p / sdp / flow に導出パラメータが復活したモデルを黙って
    # 書き出す穴が残る。**dec はここに入れない** — 有効な weight_norm を持って出荷される
    # 側で、除去は `ensure_dec_plain` が export の直前に行う。
    _assert_no_weight_norm(net_g.enc_p, net_g.sdp, net_g.dp, net_g.flow)
    return net_g, hps


def _assert_window_size(net_g: nn.Module) -> None:
    """相対位置注意の窓幅が前提どおり 4 であることを load 時に固定する（ADR 0013）。

    gather 化パッチの添字表は `clamp(rel + 4, 0, 8)` を焼き込むので、モデル側の窓幅が
    違うと**幅の違う埋め込みを gather する**。要素数は合うため shape エラーにならず、
    黙って誤った埋め込みを読む（沈黙誤値クラス）。しかも golden も同じ誤りで生成される
    ため数値突合もすり抜ける。ckpt を差し替えた瞬間に落とすのが最小の対処なので、
    ローダの側に置く。

    MUST: 走査は **net_g 全体**（先頭 1 層ずつではない）。パッチは MultiHeadAttention の
    クラス属性差し替えで、`FlowReverse` は**全 coupling × 全層**に同じ表を配る構造なので、
    層ごとに窓幅が違うモデルを先頭層だけで見ると残りの層が黙って別の幅を読む。門の粒度を
    表の配り方に揃える。
    MUST: `window_size` を持つモジュールが 1 枚も無ければ落とす（恒真化の門）。属性名が
    上流で変われば走査は静かに空振りし、以後どんな窓幅でも通る。
    """
    found = [
        (name or type(module).__name__, module.window_size)
        for name, module in net_g.named_modules()
        # 相対位置注意を使わない層は属性ごと None（実測: 窓幅を持つのは attn 層だけ）。
        if getattr(module, "window_size", None) is not None
    ]
    if not found:
        raise ValueError(
            "window_size を持つモジュールが 1 つも無い — 窓幅の門が恒真化している"
            "（相対位置注意の属性名が上流で変わった可能性）"
        )
    mismatched = [(where, size) for where, size in found if size != EXPECTED_WINDOW_SIZE]
    if mismatched:
        raise ValueError(
            f"相対位置注意の窓幅が前提（{EXPECTED_WINDOW_SIZE}）と違うモジュールが"
            f" {len(mismatched)}/{len(found)} 件: {mismatched}"
            " — 表の生成側が焼き込む幅と食い違うと誤った埋め込みを黙って読む"
        )


def _weight_norm_parameters(*modules: nn.Module) -> list[str]:
    """weight_norm 由来の導出パラメータ（`weight_g` / `weight_v` / parametrizations）の一覧。

    「残っていないこと」と「まだ残っていること」の両方の門がこの 1 本の判定を共有する
    （判定を 2 箇所に書くと必ず割れ、片方だけ緩い門になる）。
    """
    return [
        f"{type(module).__name__}.{name}"
        for module in modules
        for name, _ in module.named_parameters()
        if name.endswith(("weight_g", "weight_v")) or ".parametrizations." in name
    ]


def _assert_no_weight_norm(*modules: nn.Module) -> None:
    """weight_norm 由来の導出パラメータが無いことを固定する。

    weight_norm が残っていると `weight` は「実効重みではない」ため、そのまま IR へ
    書き出すと別のモデルになる。enc_p / sdp / dp は実測 0 件だが、無いことを前提に
    何もしていない側なので fail loudly で固定する（remove を先に通す順序制約は dec を
    扱う波で効いてくる）。
    """
    derived = _weight_norm_parameters(*modules)
    if derived:
        raise ValueError(f"weight_norm 由来のパラメータが残っている: {derived}")


def _assert_patches_not_applied(where: str) -> None:
    """参照採取がパッチ適用**前**であることを固定する（ADR 0013 の順序制約の門）。

    パッチはクラス属性のプロセス全域差し替えなので、適用後に採った「参照」はパッチ後の
    値そのものになり、同値検証が恒真化して偽 PASS する。**差が常に 0 になる**方向の
    壊れ方なので、検証が緑であること自体は何の証拠にもならない — 門でしか塞げない。
    """
    if patch.patches_applied():
        raise RuntimeError(
            f"{where}: パッチ適用後に参照値を採ろうとした — 同値検証が恒真化する"
            "（順序は「全ケースの参照を確定 → パッチ適用 → 比較」）"
        )


def _assert_weight_norm_present(module: nn.Module, where: str) -> None:
    """dec の参照採取が `remove_weight_norm` **前**であることを固定する（**逆向きの門**）。

    dec の同値検証の参照は「weight_norm が有効な原経路」で、主張の中身は
    **remove 前後のビット一致**。remove 済みの net_g で参照を採ると remove 後どうしの
    比較になり、差 0 が自明に成立して偽 PASS する。他の 3 ターゲットは「パッチが
    当たっていないこと」を要求するのに対し、ここだけは「導出パラメータが**残っている**
    こと」を要求する — 汚染源が逆向きだから。
    """
    if not _weight_norm_parameters(module):
        raise RuntimeError(
            f"{where}: remove_weight_norm 済みの net_g で参照値を採ろうとした"
            " — remove 前後の同値検証が恒真化する"
            "（順序は「全ケースの参照を確定 → remove → 比較」）"
        )


def speaker_embedding(net_g: nn.Module, speaker_id: int = SPEAKER_ID) -> torch.Tensor:
    """話者埋め込み g `[1, 512, 1]` を実重みから引く（参照実装 infer と同じ形）。

    合成乱数ではなく実重みの埋め込みを使う — g は enc_p の条件付け（cond_layer_idx 層）と
    dp / sdp の cond に入るので、実際の値域が下流の LayerNorm や spline の効き方を決める。
    golden の数値許容差の根拠を実運用の値域と対応させるための選択。
    """
    num_speakers = net_g.emb_g.num_embeddings
    if not 0 <= speaker_id < num_speakers:
        raise ValueError(f"話者 ID {speaker_id} がモデルの話者数 {num_speakers} の外")
    with torch.no_grad():
        return net_g.emb_g(torch.tensor([speaker_id], dtype=torch.int64)).unsqueeze(-1)


def style_vector(model_dir: Path, style_id: int = STYLE_ID) -> torch.Tensor:
    """スタイルベクトル `[1, 256]` を実資産（`style_vectors.npy`）から引く。

    話者埋め込みと同じ理由で合成乱数を使わない。style_proj は全音素に同じ値を足す経路
    （`[1,1,256] → [1,1,192]` の broadcast 加算）なので、値域がずれると h の平均が
    実運用と別の場所に乗る。
    """
    path = model_dir / STYLE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} が無い（ckpt と同じディレクトリに置く）")
    table = np.load(path)
    if table.ndim != 2 or table.shape[1] != 256:
        raise ValueError(f"{path} の形 {table.shape} が [スタイル数, 256] でない")
    if not 0 <= style_id < table.shape[0]:
        raise ValueError(f"スタイル ID {style_id} がスタイル数 {table.shape[0]} の外")
    return torch.from_numpy(np.ascontiguousarray(table[style_id])).to(torch.float32).unsqueeze(0)


def _generator(length: int, pad: int, salt: int) -> torch.Generator:
    """ケース（長さ・パディング）と用途ごとに独立した乱数列を引く。

    グローバル seed に依存しないので、生成順を変えても・単体で 1 ケースだけ作り直しても
    バイト一致する（golden の再生成可能性の前提）。
    """
    return torch.Generator().manual_seed(SEED + salt * 1_000_003 + length * 31 + pad)


def make_hidden(length: int) -> torch.Tensor:
    """dp 用の h `[1, 192, P]`（enc_p 出力に相当。ケースごとに別の固定 seed）。

    実 h は enc_p 側で x_mask 済みだが、ここでは**パディング列にも値を入れる** —
    dp 自身のマスク（3 箇所の `x * x_mask` と最終段）が効いていれば出力の末尾は
    厳密に 0 になり、外れれば値が漏れる。マスク経路の検出器はこの形でしか作れない。
    """
    generator = torch.Generator().manual_seed(SEED + length)
    return torch.randn(1, 192, length, generator=generator)


def make_mask(length: int, pad: int) -> torch.Tensor:
    """x_mask `[1, 1, P]` — 先頭 `P - pad` が 1、末尾 `pad` が 0 の f32。"""
    if not 0 <= pad < length:
        raise ValueError(f"pad={pad} が P={length} に対して不正")
    mask = torch.ones(1, 1, length)
    if pad:
        mask[..., length - pad :] = 0.0
    return mask


def make_noise(length: int) -> torch.Tensor:
    """sdp reverse の外部ノイズ `z_noise` `[1, 2, P]`（`noise_scale` 乗算済み）。

    MUST: 参照実装が内部で引く `torch.randn(1, 2, P) * noise_scale` と**同じ列**であること
    （同値検証はこの一致が前提）。torch の CPU 既定生成器は同一 seed なら
    `Generator` 経由でもグローバル seed 経由でも同じ列を返すので、参照側は
    `torch.manual_seed(_noise_seed(P))` を呼んでから原 forward を通す。
    """
    generator = torch.Generator().manual_seed(_noise_seed(length))
    return torch.randn(1, 2, length, generator=generator) * NOISE_SCALE


def _noise_seed(length: int) -> int:
    """`make_noise` の seed（参照側がグローバル seed として使う同じ値）。"""
    return SEED + 7 * length


def make_phones(length: int) -> torch.Tensor:
    """音素 ID `x` `[1, P]`（i64）。0 は `_`（無音記号）なので 1 以上から引く。"""
    from style_bert_vits2.nlp.symbols import SYMBOLS

    return torch.randint(1, len(SYMBOLS), (1, length), generator=_generator(length, 0, 1))


def make_tones(length: int) -> torch.Tensor:
    """アクセント ID `tone` `[1, P]`（i64）。"""
    from style_bert_vits2.nlp.symbols import NUM_TONES

    return torch.randint(0, NUM_TONES, (1, length), generator=_generator(length, 0, 2))


def make_language(length: int) -> torch.Tensor:
    """言語 ID `language` `[1, P]`（i64）。JP-Extra は日本語単言語なので全て 0。"""
    return torch.zeros((1, length), dtype=torch.int64)


def make_bert(length: int) -> torch.Tensor:
    """BERT 特徴 `bert` `[1, 1024, P]`。

    NOTE: 実 DeBERTa の hidden ではなく randn の合成値（実 hidden を作るには
    `outputs/series/deberta` の 1.3GB と音素↔文字のアライメントが要り、front の数値検証の
    目的からは遠い）。bert_proj は 1x1 conv 1 本で下流は LayerNorm 群なので、
    値域の違いは h のスケールに一様に効くだけで、誤差の伸び方の構造は変わらない。
    """
    return torch.randn(1, 1024, length, generator=_generator(length, 0, 3))


def build_cases(
    g: torch.Tensor,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
) -> tuple[tuple[str, dict[str, torch.Tensor]], ...]:
    """dp の golden ケース `(名前, {入力名: テンソル})`。キーはグラフ入力名に一致させる。"""
    return tuple(
        (name, {"h": make_hidden(length), "x_mask": make_mask(length, pad), "g": g})
        for name, length, pad in cases
    )


def front_inputs(length: int, pad: int, g: torch.Tensor, style: torch.Tensor) -> dict[str, Any]:
    """front の入力一式（グラフ入力名 + 参照実装が要る `x_lengths`）。

    x / tone / bert は**パディング列にも値を入れる**（dp の h と同じ理由 — マスク経路が
    効いていれば出力の末尾は厳密に 0 になる）。
    """
    return {
        "x": make_phones(length),
        "x_mask": make_mask(length, pad),
        "tone": make_tones(length),
        "language": make_language(length),
        "bert": make_bert(length),
        "style_vec": style,
        "g": g,
        "z_noise": make_noise(length),
        "x_lengths": torch.tensor([length - pad], dtype=torch.int64),
    }


def make_latent(length: int, salt: int) -> torch.Tensor:
    """flow / dec の入力になる潜在系列 `[1, 192, T]`（ケース・用途ごとに別の固定 seed）。

    実チェーンの `z_p` は `m_p + randn·exp(logs_p)·noise_scale` で、値域は O(1) の
    正規分布そのもの。合成 randn を使うのはそのため（front の `bert` と違い、実分布から
    大きく外れない）。
    """
    return torch.randn(1, 192, length, generator=_generator(length, 0, salt))


def flow_inputs(length: int, pad: int, g: torch.Tensor) -> dict[str, torch.Tensor]:
    """flow / voice の入力一式（グラフ入力名がキー）。

    `z_p` は**パディング列にも値を入れる**（dp / front と同じ理由 — `y_mask` 乗算が効いて
    いれば出力の末尾は厳密に 0 になり、外れれば値が漏れる）。表 `idx_k` / `valid` は
    `sbv2.patch.build_relattn_tables`（ホスト TS 鏡像の Python 側正本）から採る。
    """
    idx_k, valid = patch.build_relattn_tables(length, EXPECTED_WINDOW_SIZE)
    return {
        "z_p": make_latent(length, salt=4),
        "y_mask": make_mask(length, pad),
        "g": g,
        "idx_k": idx_k,
        "valid": valid,
    }


def dec_inputs(length: int, pad: int, g: torch.Tensor) -> dict[str, torch.Tensor]:
    """dec の入力一式。

    dec には**マスク入力が無い**（`Generator.forward(x, g)`）ので、`padded` ケースは実
    チェーンと同じく**入力側**で末尾を 0 にする（融合 voice が `dec(z * y_mask, g)` を
    呼ぶのと同じ形）。dec 単体にマスク経路は存在しないため、このケースが踏むのは
    「末尾が 0 の入力でも上流と同じ数が出る」ことだけで、front / flow の padded のような
    漏れ検出器にはならない — 検出器は voice 側が持つ。
    """
    return {"x": make_latent(length, salt=5) * make_mask(length, pad), "g": g}


def build_flow_cases(
    g: torch.Tensor,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
) -> tuple[tuple[str, dict[str, torch.Tensor]], ...]:
    """flow / voice の golden ケース（入力は共通 — 融合の有無だけが違う）。"""
    return tuple((name, flow_inputs(length, pad, g)) for name, length, pad in cases)


def build_dec_cases(
    g: torch.Tensor,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
) -> tuple[tuple[str, dict[str, torch.Tensor]], ...]:
    """dec の golden ケース。"""
    return tuple((name, dec_inputs(length, pad, g)) for name, length, pad in cases)


def build_front_cases(
    g: torch.Tensor,
    style: torch.Tensor,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
) -> tuple[tuple[str, dict[str, torch.Tensor]], ...]:
    """front の golden ケース。`x_lengths` は参照専用なので golden には載せない。"""
    return tuple(
        (
            name,
            {
                key: value
                for key, value in front_inputs(length, pad, g, style).items()
                if key != "x_lengths"
            },
        )
        for name, length, pad in cases
    )


def _write_io(
    module: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, dict[str, torch.Tensor]]],
    out_dir: Path,
) -> list[str]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    forward は**グラフ入力の宣言順**で呼ぶ（IR の入力名 = forward の引数名なので、
    ケース辞書のキーが 1 つでも食い違えば KeyError で落ちる — 名前の対応が黙ってずれた
    まま別の引数に流れ込む形を潰す）。
    """
    written: list[str] = []
    for name, args in cases:
        with torch.no_grad():
            output = module(*(args[declared.name] for declared in graph.inputs))
        outputs = (output,) if isinstance(output, torch.Tensor) else tuple(output)
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{name}: eager 出力 {len(outputs)} 本が IR 出力 {len(graph.outputs)} 本と違う"
            )
        # MUST: io も IR の意味論 dtype の実表現へ落とす（ADR 0009 の境界正規化）。
        tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name} の入力 '{declared.name}'"
            )
            for declared in graph.inputs
        }
        for index, value in enumerate(outputs):
            tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
                value.detach().contiguous(), f"{name} の出力 {index}"
            )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def _module_name(weight_fqn: str) -> str:
    """`iter_quant_targets` が返す重み FQN（`<module>.weight` / 根なら `weight`）→ モジュール FQN。

    `include` 述語（`fake_quant_int8` / `fake_quant_int4`）が見るのは**モジュール**の FQN で、
    台帳のキーは**重み**の FQN — 2 つの空間を取り違えると述語が 1 本も当たらない。
    `removesuffix` 1 発で書けないのは根モジュール（`""`）だけが `.` を持たないから。
    """
    return weight_fqn[: -len(".weight")] if weight_fqn.endswith(".weight") else ""


def _i4_module_names(module: nn.Module) -> frozenset[str]:
    """i4 group32 で丸めるモジュールの FQN 集合（混成の**排他割り**の唯一の源）。

    適格は 3 条件の積: ① {@link I4_MODULE_TYPES} の型であること（emit の i4 適格 =
    `emit.I4_WEIGHT_OPS` の重みスロット限定 — ADR 0069 決定 5 とその conv1d 追補。外れた
    テンソルへ i4 を明示指定すると emit が fail loudly する）② `groups == 1`（conv1d の i4 は
    igemm 変種だけが展開でき、depthwise〈`DDSConv` の `convs_sep`〉は direct カーネル =
    i4 非対応）③ 平坦化後の行長が group 長で割り切れること（i4 は端数 group を作らない MUST —
    同決定 2。外れた重みは**構成ごと落とすのではなく対象から外す**〈`measure_quant`
    `census_w4_targets` と同じ扱い〉ので、非適格の linear / conv1d は i8 側へ落ちる）。

    対象列挙を core（`iter_quant_targets`）に通すのは、丸めが見る集合とここが数える集合を
    1 本の実装のままにするため。i8 側の述語を「この集合に居ない」で書くのも同じ理由で、
    2 つの述語を別々の綴りから作るとどちらにも入らない重みが**黙って f32 のまま残る**
    （二重丸め禁止の逆側の穴で、値は正しいままサイズだけが戻る）。

    NOTE: `nn.ConvTranspose1d` は `nn.Conv1d` の派生ではないので、型で自然に落ちる
    （`dec` の `ups` — 行軸が先頭でない重みは pack 順が合わず i4 にできない）。
    """
    modules = dict(module.named_modules())
    names: set[str] = set()
    for fqn, weight, axis in iter_quant_targets(module, I4_MODULE_TYPES):
        name = _module_name(fqn)
        if getattr(modules[name], "groups", 1) != 1:
            continue
        if channel_rows(weight, axis).shape[-1] % DEFAULT_GROUP_SIZE == 0:
            names.add(name)
    return frozenset(names)


def _fake_quant(
    dtype: str, module: nn.Module, target: str
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """格納 dtype の表現可能値へ**実効重み**を丸め、scale 台帳と 1 本単位の格納指定を返す。

    ADR 0006 / 0018（f16）/ 0019（i8）/ 0069（i4）の fake-quant。台帳のキーはモデル内 FQN で、
    emit 側が safetensors のテンソルキーとして突き合わせる（`id()` 突合は禁止 — ADR 0006）。

    i4 系列は混成（適格な linear / conv1d = i4 group32・残り = i8 per-channel）で、2 つの述語は
    {@link _i4_module_names} から**排他に**割る（`quantize.py` の混成 MUST）。返す override は
    「i4 の scale 台帳のキー全部を i4 に振る」写像で、emit 側は明示指定を満たせなければ
    fail loudly する（`emit._plan_i4` — 沈黙 i8 へ落ちる経路は無い）。

    MUST: 当てる相手は **export する `nn.Module` そのもの**（`net_g` 全体ではない）。
    そのターゲットのグラフに現れない重みまで動かすと、同じ net_g から別ターゲットを
    採るときの前提が濁る。

    MUST（順序）: ① `remove_weight_norm` / パッチ適用の**後**に呼ぶ。remove より先に丸めると
    `weight_g` / `weight_v` を丸めることになり、そこから作られる実効重みは f16 / i8 の格子に
    乗らない（`ensure_dec_plain` の順序制約そのもの）。i8 は**さらに strict** で、捨てられる
    要素が amax に効くと per-channel scale そのものがずれる。② 参照・golden の採取の**前**に
    呼ぶ（`quantize` モジュールの MUST — 後に当てると golden だけが元の重みで計算され、E2E の
    差に量子化誤差が混ざって tolerance の意味が消える）。

    5 ターゲットはそれぞれ `load_net_g` で新しい net_g を組むので、丸めがターゲットを跨いで
    漏れることはない。voice は flow と dec を 1 つのラッパで束ねるためここ 1 回で両方に
    掛かる（丸めは f16 / i8 とも冪等 — i8 は ±127 に閉じた量子化が f32 の不動点になる
    〈ADR 0019〉。ただし依存しているのは冪等性ではなく「実効重みが確定した後に呼ぶ」という
    上の順序で、実重みでの冪等性は `sbv2/tests/test_export.py` が固定する）。

    NOTE: `g`（`emb_g` の話者埋め込み）と `style_vec` は export 対象のモジュールに含まれない
    **グラフ入力**なので丸めない — 入力は参照と GPU が同じバイト列を読む側で、格納 dtype の
    影響を受けない。front の焼き込み相対位置表も同じくモジュールの重みではないので触れない
    （グラフ定数は適格外 = f32 格納 — emit.py の適格判定）。
    """
    if dtype == "f32":
        return {}, {}
    if dtype == "i8":
        int8 = fake_quant_int8(module)
        print(
            f"[fake-quant] {target}: i8 per-channel へ丸めた — {int8.describe()}",
            flush=True,
        )
        return int8.scales, {}
    if dtype == "f16":
        report = round_weights_to_f16(module)
        print(f"[fake-quant] {target}: f16 表現可能値へ丸めた — {report.describe()}", flush=True)
        return {}, {}
    i4_names = _i4_module_names(module)
    i8_names = {
        _module_name(fqn) for fqn, _, _ in iter_quant_targets(module, QUANT_MODULE_TYPES)
    } - i4_names
    int4 = fake_quant_int4(
        module, DEFAULT_GROUP_SIZE, include=lambda name: name in i4_names, op_types=I4_MODULE_TYPES
    )
    # MUST: 排他割りの**両側を列挙してから**丸める。i8 側を「i4 に居ない」の否定だけで書くと、
    # 全ての量子化対象が i4 適格な系列（`flow` は conv1d だけで構成され、全本が groups == 1 かつ
    # 整除する）で `fake_quant_int8` の「対象 0 本」が発火して export が落ちる — 0 本が
    # **数え上げの結果**である以上、沈黙ではないので丸めごと飛ばす。
    i8_scales: Mapping[str, torch.Tensor] = {}
    if i8_names:
        rest = fake_quant_int8(module, include=lambda name: name in i8_names)
        i8_scales = rest.scales
        remainder = f"残りは i8 per-channel — {rest.describe()}"
    else:
        remainder = "i8 側は対象 0 本（量子化対象が全て i4 適格）"
    print(
        f"[fake-quant] {target}: 適格 linear / conv1d を i4 group へ丸めた —"
        f" {int4.describe()} / {remainder}",
        flush=True,
    )
    return {**i8_scales, **int4.scales}, dict.fromkeys(int4.scales, "i4")


def _summary(
    target: str,
    out_dir: Path,
    hps: Any,
    sym_max: int,
    graph: IrGraph,
    written: Sequence[str],
    cases: Sequence[tuple[str, int, int]],
    elapsed: float,
    dtype: str,
) -> dict[str, Any]:
    # 適格 = 圧縮のまま GPU 常駐する重み / 適格外 = f32 格納（ADR 0006 の常設診断）。
    breakdown = storage_breakdown(graph)
    return {
        "target": target,
        "dir": str(out_dir),
        "dtype": dtype,
        "version": hps.version,
        "sym_max": sym_max,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        # i8 の companion scale（ADR 0019）。f32 / f16 では 0。
        "scale_bytes": breakdown.scale_bytes,
        "model_bytes": (out_dir / MODEL_FILE).stat().st_size,
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": list(written),
        "case_lengths": {name: length for name, length, _ in cases},
        "seconds": round(elapsed, 1),
    }


def export_dp(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """dp の IR コンテナと golden io を書き、要約を返す。"""
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    module = DurationPredictorGraph(net_g.dp)
    # 前処理は無い（パッチも remove_weight_norm も通らない）ので、ここが「実効重み確定後」。
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_DP)
    g = speaker_embedding(net_g)
    built = build_cases(g, cases)

    # 例示入力は padded ケース（x_mask に 0 を含む形）。min=2 は 0/1 特殊化を避けるため、
    # max は宣言そのもの（ADR 0010 — 畳み込みの評価点は range_constraints から取る）。
    example = dict(built[-1][1])
    phonemes = Dim("P", min=2, max=sym_max)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = export_to_file(
        module,
        (example["h"], example["x_mask"], example["g"]),
        out_dir / MODEL_FILE,
        dynamic_shapes=({2: phonemes}, {2: phonemes}, {}),
        symbol_names=("P",),
        weight_dtype=BASE_WEIGHT_DTYPES[dtype],
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
    )
    written = _write_io(module, graph, built, out_dir)
    return _summary(
        TARGET_DP,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def export_front(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """front（enc_p + dp + sdp reverse の融合グラフ）の IR コンテナと golden io を書く。

    パッチ層をここで当てる — golden は**パッチ適用後**の eager 出力（= IR が計算すべき数の
    正）で、パッチ前の参照実装との差は `--verify` が別プロセスで実測する二層構造
    （ADR 0013）。
    """
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    patch.apply_all_patches()
    module = patch.Sbv2Front(net_g)
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_FRONT)
    built = build_front_cases(speaker_embedding(net_g), style_vector(model_dir), cases)

    # 例示入力は padded ケース（x_mask に 0 を含む形）。
    example = dict(built[-1][1])
    phonemes = Dim("P", min=2, max=sym_max)
    # 動的軸: x / tone / language は [1,P]、x_mask / bert / z_noise は [...,P]。
    # style_vec [1,256] と g [1,512,1] は P に依らない。
    dynamic_shapes = (
        {1: phonemes},
        {2: phonemes},
        {1: phonemes},
        {1: phonemes},
        {2: phonemes},
        {},
        {},
        {2: phonemes},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = export_to_file(
        module,
        tuple(example[declared] for declared in FRONT_INPUT_ORDER),
        out_dir / MODEL_FILE,
        dynamic_shapes=dynamic_shapes,
        symbol_names=("P",),
        weight_dtype=BASE_WEIGHT_DTYPES[dtype],
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
    )
    written = _write_io(module, graph, built, out_dir)
    return _summary(
        TARGET_FRONT,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def ensure_dec_plain(net_g: nn.Module) -> None:
    """dec の weight_norm を除去して実効重みに畳む（冪等）。

    weight_norm が残っている間、`Conv1d.weight` は**実効重みではない**（`weight_g` /
    `weight_v` から forward pre-hook が毎回作る）。そのまま IR へ書き出すと別のモデルに
    なるので、export 前に必ず通す。冪等なのは融合 voice が dec と同じ net_g を共有する
    ため（二重適用は `remove_weight_norm` が RuntimeError で落ちる）。

    MUST（順序制約 — ADR 0013 / 0018）: 重みの丸め（`_fake_quant`）は **remove 後の実効重み**
    に当てる。remove より先に丸めると `weight_g`/`weight_v` を丸めることになり、実効重みは
    丸めの格子に乗らない（配信サイズは減るが数値は別物になる）。dec / voice の export は
    どちらもこの関数の**後**に `_fake_quant` を呼ぶ形で並べてある。

    数値同値（remove 前後）は `--verify dec` が実重み・全ケースで実測する。
    """
    if any(hasattr(module, "weight_g") for module in net_g.dec.modules()):
        net_g.dec.remove_weight_norm()
    _assert_no_weight_norm(net_g.dec)


def export_flow(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = FLOW_SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """flow（TransformerCouplingBlock reverse）の IR コンテナと golden io を書く。

    相対位置注意の `(T,T)` 表は**グラフ入力**（`idx_k` / `valid`）。front の焼き込み方式と
    違うのは sym_max が桁違いだから（4096 で O(T²) = 134MB — ADR 0013）。
    """
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    patch.apply_all_patches()
    module = patch.FlowReverse(net_g)
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_FLOW)
    built = build_flow_cases(speaker_embedding(net_g), cases)

    example = dict(built[-1][1])  # padded ケース（y_mask に 0 を含む形）
    frames = Dim("T", min=2, max=sym_max)
    # z_p [1,192,T] / y_mask [1,1,T] / g [1,512,1] / idx_k [T,T] / valid [T,T]。
    dynamic_shapes = (
        {2: frames},
        {2: frames},
        {},
        {0: frames, 1: frames},
        {0: frames, 1: frames},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = export_to_file(
        module,
        tuple(example[declared] for declared in FLOW_INPUT_ORDER),
        out_dir / MODEL_FILE,
        dynamic_shapes=dynamic_shapes,
        symbol_names=("T",),
        weight_dtype=BASE_WEIGHT_DTYPES[dtype],
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
    )
    written = _write_io(module, graph, built, out_dir)
    return _summary(
        TARGET_FLOW,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def export_dec(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = FLOW_SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """dec（HiFi-GAN Generator）の IR コンテナと golden io を書く。

    パッチ層は要らない（注意も spline も通らない）。前処理は `remove_weight_norm` だけで、
    ラッパも置かない — `Generator.forward(x, g)` の引数名がそのまま IR の入力名になる。
    """
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    ensure_dec_plain(net_g)
    module = net_g.dec
    # MUST: remove_weight_norm の**後**（`ensure_dec_plain` が予告している順序制約）。
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_DEC)
    built = build_dec_cases(speaker_embedding(net_g), cases)

    example = dict(built[-1][1])
    frames = Dim("T", min=2, max=sym_max)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = export_to_file(
        module,
        tuple(example[declared] for declared in DEC_INPUT_ORDER),
        out_dir / MODEL_FILE,
        dynamic_shapes=({2: frames}, {}),
        symbol_names=("T",),
        weight_dtype=BASE_WEIGHT_DTYPES[dtype],
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
    )
    written = _write_io(module, graph, built, out_dir)
    return _summary(
        TARGET_DEC,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def export_voice(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = FLOW_SYM_MAX,
    cases: Sequence[tuple[str, int, int]] = GOLDEN_CASES,
    dtype: str = "f32",
) -> dict[str, Any]:
    """voice（flow reverse + dec の融合 1 グラフ）の IR コンテナと golden io を書く。

    **このターゲットの E2E が緑になった時点で SBV2 の全チェーンが成立する**（front で
    durations を出し、ホスト側で z_p を組み、ここで波形になる）。融合の利得は中間 z の
    readback 往復の排除で、代わりに z のデバッグ突合は flow 単体側でしかできない。
    """
    started = time.perf_counter()
    net_g, hps = load_net_g(model_dir)
    patch.apply_all_patches()
    ensure_dec_plain(net_g)
    module = patch.Sbv2Voice(net_g)
    # MUST: remove_weight_norm の**後**（dec を内包するので dec 単体と同じ順序制約）。
    scales, dtype_overrides = _fake_quant(dtype, module, TARGET_VOICE)
    built = build_flow_cases(speaker_embedding(net_g), cases)

    example = dict(built[-1][1])
    frames = Dim("T", min=2, max=sym_max)
    dynamic_shapes = (
        {2: frames},
        {2: frames},
        {},
        {0: frames, 1: frames},
        {0: frames, 1: frames},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = export_to_file(
        module,
        tuple(example[declared] for declared in FLOW_INPUT_ORDER),
        out_dir / MODEL_FILE,
        dynamic_shapes=dynamic_shapes,
        symbol_names=("T",),
        weight_dtype=BASE_WEIGHT_DTYPES[dtype],
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
    )
    written = _write_io(module, graph, built, out_dir)
    return _summary(
        TARGET_VOICE,
        out_dir,
        hps,
        sym_max,
        graph,
        written,
        cases,
        time.perf_counter() - started,
        dtype,
    )


def reference_front_outputs(net_g: nn.Module, inputs: dict[str, Any]) -> list[torch.Tensor]:
    """**パッチ前**の参照実装で front の 4 出力を得る。

    MUST: パッチ適用前にだけ呼べる。パッチ後に呼ぶと参照そのものがパッチ後の値になり、
    同値検証が恒真化して偽 PASS する（ADR 0013）。
    """
    _assert_patches_not_applied("front の参照採取")
    with torch.no_grad():
        h, m_p, logs_p, x_mask = net_g.enc_p(
            inputs["x"],
            inputs["x_lengths"],
            inputs["tone"],
            inputs["language"],
            inputs["bert"],
            inputs["style_vec"],
            g=inputs["g"],
        )
        # 参照実装は x_lengths から x_mask を組む。こちらが外から渡す x_mask と食い違うと
        # 「同じ入力の比較」でなくなるので、ここで固定する。
        if not torch.equal(x_mask, inputs["x_mask"]):
            raise AssertionError("参照実装が生成した x_mask が golden の x_mask と違う")
        # 参照の reverse 経路は内部で randn を 1 回だけ引く。同じ seed をグローバルへ置けば
        # make_noise（Generator 経由）と同じ列になる（CPU 既定生成器の同一性）。
        torch.manual_seed(_noise_seed(int(inputs["x"].shape[1])))
        logw_sdp = net_g.sdp(h, x_mask, g=inputs["g"], reverse=True, noise_scale=NOISE_SCALE)
        logw_dp = net_g.dp(h, x_mask, g=inputs["g"])
    return [logw_sdp, logw_dp, m_p, logs_p]


def _diff_entry(
    length: int,
    pad: int,
    names: Sequence[str],
    got: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """1 ケース分の同値レポート（出力名 → maxdiff と、全出力のビット一致）。

    `bit_exact` は「差 0」より強い主張（`0.0 == -0.0` は差 0 だがビットは違う）で、
    remove_weight_norm のように**ビット一致が主張の中身**である検証で意味を持つ。
    """
    return {
        "length": length,
        "pad": pad,
        "maxdiff": {
            name: float((a - b).abs().max())
            for name, a, b in zip(names, got, expected, strict=True)
        },
        "bit_exact": all(torch.equal(a, b) for a, b in zip(got, expected, strict=True)),
    }


def verify_front(
    model_dir: Path, *, cases: Sequence[tuple[int, int]] = VERIFY_CASES
) -> list[dict[str, Any]]:
    """front のパッチ前後 eager 同値を実重みで実測する。

    MUST: 順序は「**全ケースの参照値を確定** → パッチ適用 → 比較」。パッチはクラス属性の
    プロセス全域差し替えなので、1 ケースずつ「参照 → 比較」を回すと 2 ケース目以降の参照が
    パッチ後の値になる（差が常に 0 になる恒真化）。
    """
    net_g, _hps = load_net_g(model_dir)
    g = speaker_embedding(net_g)
    style = style_vector(model_dir)

    references = {
        (length, pad): reference_front_outputs(net_g, front_inputs(length, pad, g, style))
        for length, pad in cases
    }

    patch.apply_all_patches()
    front = patch.Sbv2Front(net_g)

    report: list[dict[str, Any]] = []
    for length, pad in cases:
        inputs = front_inputs(length, pad, g, style)
        with torch.no_grad():
            outputs = front(*(inputs[name] for name in FRONT_INPUT_ORDER))
        report.append(
            _diff_entry(length, pad, FRONT_OUTPUT_NAMES, outputs, references[(length, pad)])
        )
    return report


def verify_flow(
    model_dir: Path, *, cases: Sequence[tuple[int, int]] = FLOW_VERIFY_CASES
) -> list[dict[str, Any]]:
    """flow: 表入力版 `FlowReverse` と参照 `TransformerCouplingBlock(reverse=True)` の同値。

    MUST: front と同じ順序制約（参照を全部採ってからパッチ）。MHA パッチはクラス属性
    差し替えなので `net_g.flow` 側の注意層にも同時に効く。
    """
    net_g, _hps = load_net_g(model_dir)
    g = speaker_embedding(net_g)

    _assert_patches_not_applied("flow の参照採取")
    references = {}
    for length, pad in cases:
        inputs = flow_inputs(length, pad, g)
        with torch.no_grad():
            references[(length, pad)] = net_g.flow(
                inputs["z_p"], inputs["y_mask"], g=inputs["g"], reverse=True
            )

    patch.apply_all_patches()
    flow_rev = patch.FlowReverse(net_g)

    report: list[dict[str, Any]] = []
    for length, pad in cases:
        inputs = flow_inputs(length, pad, g)
        with torch.no_grad():
            got = flow_rev(*(inputs[name] for name in FLOW_INPUT_ORDER))
        report.append(_diff_entry(length, pad, ("z",), (got,), (references[(length, pad)],)))
    return report


def verify_dec(
    model_dir: Path, *, cases: Sequence[tuple[int, int]] = FLOW_VERIFY_CASES
) -> list[dict[str, Any]]:
    """dec: `remove_weight_norm` **前後**の eager 同値（参照 = weight_norm 有効の原経路）。

    recon はビット一致を 1 ケース（z=(1,192,50)）でしか実測しておらず、weight_norm の
    実効重み `g · v / ‖v‖` が f32 で厳密に再現される保証はスペックには無い。ここで**全
    ケース**を突合してその未検証事項を閉じる（`bit_exact` が主張の中身）。

    MUST: 参照を全ケース採ってから remove する。`remove_weight_norm` はパラメータを
    破壊的に畳むので、1 ケースずつ回すと 2 ケース目以降の参照が remove 後の値になる。
    """
    net_g, _hps = load_net_g(model_dir)
    g = speaker_embedding(net_g)

    _assert_weight_norm_present(net_g.dec, "dec の参照採取")
    references = {}
    for length, pad in cases:
        inputs = dec_inputs(length, pad, g)
        with torch.no_grad():
            references[(length, pad)] = net_g.dec(inputs["x"], g=inputs["g"])

    ensure_dec_plain(net_g)

    report: list[dict[str, Any]] = []
    for length, pad in cases:
        inputs = dec_inputs(length, pad, g)
        with torch.no_grad():
            got = net_g.dec(inputs["x"], g=inputs["g"])
        report.append(_diff_entry(length, pad, ("audio",), (got,), (references[(length, pad)],)))
    return report


def verify_voice(
    model_dir: Path, *, cases: Sequence[tuple[int, int]] = FLOW_VERIFY_CASES
) -> list[dict[str, Any]]:
    """voice: 融合ラッパと参照チェーン（未パッチ flow + weight_norm 有効 dec）の同値。

    flow と dec の**両方**の汚染源を跨ぐので、このプロセスでは他の検証を一切走らせない
    （CLI が 1 プロセス 1 検証を強制する）。
    """
    net_g, _hps = load_net_g(model_dir)
    g = speaker_embedding(net_g)

    # 汚染源が 2 つ（パッチと remove）あるので門も 2 枚。片方だけ置くと、もう片方の順序
    # 違反が「差 0 の偽 PASS」として通る。
    _assert_patches_not_applied("voice の参照採取")
    _assert_weight_norm_present(net_g.dec, "voice の参照採取")
    references = {}
    for length, pad in cases:
        inputs = flow_inputs(length, pad, g)
        with torch.no_grad():
            z = net_g.flow(inputs["z_p"], inputs["y_mask"], g=inputs["g"], reverse=True)
            references[(length, pad)] = net_g.dec(z * inputs["y_mask"], g=inputs["g"])

    patch.apply_all_patches()
    ensure_dec_plain(net_g)
    voice = patch.Sbv2Voice(net_g)

    report: list[dict[str, Any]] = []
    for length, pad in cases:
        inputs = flow_inputs(length, pad, g)
        with torch.no_grad():
            got = voice(*(inputs[name] for name in FLOW_INPUT_ORDER))
        report.append(_diff_entry(length, pad, ("audio",), (got,), (references[(length, pad)],)))
    return report


EXPORTERS = {
    TARGET_DP: export_dp,
    TARGET_FRONT: export_front,
    TARGET_FLOW: export_flow,
    TARGET_DEC: export_dec,
    TARGET_VOICE: export_voice,
}

#: `--verify <target>` → 検証関数。**dp は無い** — パッチも前処理も通らないので「参照」と
#: 「対象」が同一のモジュールになり、比較が恒真になる。
VERIFIERS = {
    TARGET_FRONT: verify_front,
    TARGET_FLOW: verify_flow,
    TARGET_DEC: verify_dec,
    TARGET_VOICE: verify_voice,
}

#: 検証レポートの表示に使う記号名（front 系は P = 音素数、flow 系は T = フレーム数）。
VERIFY_SYMBOL = {
    TARGET_FRONT: "P",
    TARGET_FLOW: "T",
    TARGET_DEC: "T",
    TARGET_VOICE: "T",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は --dtype ごとの系列 —"
        " outputs/series/sbv2-<--model-dir のディレクトリ名>{,-f16,-i8,-i4}/）",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help="重みの格納 dtype（f16 / i8 は fake-quant してから適格スロットだけ圧縮格納する"
        " — ADR 0018 / 0019。i4 は混成で、適格 linear / conv1d だけ group32 の i4・"
        "残りは i8 — ADR 0069。**emit 専用**で"
        " --verify とは併用できない）",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        default=None,
        help=f"emit するターゲット（繰り返し可・既定は全て: {', '.join(TARGETS)}）",
    )
    parser.add_argument(
        "--sym-max",
        type=int,
        default=None,
        help=(
            "記号次元の上限（既定はターゲット別: "
            + " / ".join(f"{name}={value}" for name, value in TARGET_SYM_MAX.items())
            + "）。逸脱は単一 --target のときだけ許す"
        ),
    )
    parser.add_argument(
        "--verify",
        choices=sorted(VERIFIERS),
        default=None,
        help="参照実装との eager 同値を検証する（emit はしない・1 プロセス 1 ターゲット）",
    )
    args = parser.parse_args(argv)

    # MUST: 同一プロセスでの併用を機械的に拒否する。emit はパッチを当て・weight_norm を
    # 畳むので、後から採る「前の参照」が既に汚染済みになり、同値検証が恒真化して偽 PASS
    # する（ADR 0013）。順序を main の中で気をつける規律にはしない — 規律は破れるが、
    # ここで落とせば破れない。
    if args.verify is not None and args.target is not None:
        parser.error(
            "--verify と --target は同一プロセスで併用できない"
            "（パッチのプロセス全域差し替えで参照が汚染され、同値検証が恒真化する）"
        )

    if args.verify is not None:
        if args.sym_max is not None:
            parser.error("--sym-max は emit 専用（検証は eager 実行で記号次元を持たない）")
        # MUST: --dtype も emit 専用。検証は格納形式を一切見ない eager 比較で、丸めを
        # 足しても主張（パッチ前後の同値 / remove 前後のビット一致）は変わらない。むしろ
        # dec / voice では**当てられない** — 丸めは remove の後にしか置けないのに参照は
        # remove の前に採るので、丸めた側と丸めていない側の比較になって bit_exact が
        # 常に False へ落ちる（門が壊れたのか資産が壊れたのか区別できなくなる）。
        if args.dtype != "f32":
            parser.error(
                f"--dtype {args.dtype} は emit 専用（--verify とは併用できない）— 検証は"
                "格納形式を見ない eager 比較で、dec / voice では丸めを remove_weight_norm の"
                "後にしか当てられないのに参照は remove の前に採るため、"
                "丸めた側と丸めていない側の比較になって bit_exact の主張が壊れる"
            )
        report = VERIFIERS[args.verify](args.model_dir)
        symbol = VERIFY_SYMBOL[args.verify]
        for entry in report:
            diffs = " ".join(f"{k}={v:.3e}" for k, v in entry["maxdiff"].items())
            print(
                f"{symbol}={entry['length']:4d} pad={entry['pad']}:"
                f" {diffs} bit_exact={entry['bit_exact']}"
            )
        worst = max(max(entry["maxdiff"].values()) for entry in report)
        print(f"worst maxdiff = {worst:.3e}")
        print(f"bit_exact all = {all(entry['bit_exact'] for entry in report)}")
        return

    targets = list(dict.fromkeys(args.target if args.target is not None else TARGETS))
    # MUST: 1 つの --sym-max を複数ターゲットへ配ると、既定が桁違いに違う front 系と
    # flow 系のどちらかが必ず誤値になる（しかも沈黙する — ADR 0013）。
    if args.sym_max is not None and len(targets) != 1:
        parser.error("--sym-max は --target をちょうど 1 つ指定したときだけ使える")
    if args.out is None:
        args.out = default_out_root(args.model_dir, args.dtype)

    summaries = [
        EXPORTERS[target](
            args.model_dir,
            args.out / target,
            sym_max=args.sym_max if args.sym_max is not None else TARGET_SYM_MAX[target],
            dtype=args.dtype,
        )
        for target in targets
    ]
    print(json.dumps(summaries, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
