"""軽量な母音認識 CRNN（音声 → リップシンク用の母音クラス列）を IR v1 コンテナ +
golden io へ書き出す台本。**長さは記号軸なので、グラフは 1 本だけ**。

    uv run python -m vowel_detector.export
    uv run python -m vowel_detector.export --verify  # nn.GRU との同値（差し替えの A/B）

生成物は `outputs/series/` 配下で、リポジトリ直下の `.gitignore` によりコミット対象外。

## 依存グループを足さない理由（モデル定義を逐語で写す）

追加の依存グループは**要らない**（torch は基本依存）。上流のモデル定義
`training/src/vowel_detector/crnn.py` は 20 行の `nn.Module` だが、import すると
`vowel_detector.phonemes` 経由で pyopenjtalk（OpenJTalk のネイティブビルド）を、
`vowel_detector.features` 経由で librosa を引き込む — どちらも G2P と特徴抽出のためのもので、
この台本には 1 行も要らない。したがって**モデル定義はここへ逐語で写す**（{@link Crnn}）。

写しが上流とずれる危険は `load_state_dict(strict=True)` が受け持つ: 重みは 22 本の
state_dict なので、層の並び・名前・形のどれが食い違っても読み込みが落ちる（値だけが静かに
変わる書き換えは、モデル定義の**構造**の写しでは起こせない）。

## 長さ軸（記号 `T` = 20ms フレーム数）

グラフ入力は 10ms フレーム列 `features f32 [1, 2T, 83]`、出力は 20ms フレームの
ロジット `[1, T, 8]`（conv の stride 2 が時間を半分に畳む）。

**記号は入力側ではなく出力側の格子に置く**（`2*Dim("T")`）。素の `Dim("T")` を入力へ置くと
conv1d(stride=2, pad=2, k=5) の出力 extent が `((T−1)//2)+1` の**床除算**になり、
karume の次元言語（`coeff·sym+offset` — ADR 0010）に載らない。`2*Dim("T")` なら出力 extent は
ちょうど `T` のアフィン式になる。

その帰結として **グラフ入力の 10ms フレーム数は常に偶数**（= `2T`）がランタイム契約になる。
奇数フレームは**末尾 1 本を切り捨てて**渡す（`packages/models/src/vowel-detector/pipeline.ts`
と実重み E2E が既に採っている規約 — 半端フレームは入力 1 本分しか持たない出力になり、
後処理の「1 フレーム = 20ms」が末尾だけ崩れる）。

記号を通せるのは `nn.GRU` を `karume::gru_scan` へ差し替えているから（下の「差し替え層」）。
`aten.gru.input` の分解は `run_decompositions` が**時間方向へ完全展開**するので、
`Dim("T")` は `Specializations unexpectedly required (T)` で落ちる（T10=200 で 8,434 ノード）。
差し替え後は **T 非依存の 19 ノード**で、長さバケットも右ゼロ pad も要らない。

`--length` は emit する golden ケースと**トレースの例**の長さで、グラフの受理集合ではない
（**2 の倍数**だけを受けるのは上と同じ理由）。

## 差し替え層（`vowel_detector/patch.py`）

`Crnn.forward` は `nn.GRU.forward` を通さず、双方向 2 層を**単方向 1 層 × 4 本の
`gru_scan{,_reverse}` + `cat` 2 本**へ割る（入力側 GEMM は素の `linear`）。差し替えが
`nn.GRU` と**ビット一致**することは `vowel_detector/tests/test_patch.py` が実測で固定し、
実重みでは `--verify`（複数長）と emit ごとの常設門（{@link _write_io}）が毎回踏む。

`nn.GRU` モジュール自体は**残す** — `load_state_dict(strict=True)` が受け持つ写しの検査
（`weight_ih_l{k}` / `bias_hh_l{k}_reverse` … の 22 本）はパラメータの所属ごと変えると
効かなくなる。差し替えるのは forward の経路だけ。

MUST: 呼び出し側は**右ゼロ pad で長さを合わせない**。逆方向 GRU が pad 側から状態を持ち
帰るので、T_true=137 を T=500 まで右ゼロ pad した実測で max abs diff 5.91 / argmax 一致率
0.971（誤差は末尾に集中 — 先頭 0.138 / 末尾 5.915）。{@link build_cases} の `voiced` を
T10=138 → 500 で pad しても同型（max abs diff 5.50 / 先頭 0.59 / 末尾 5.50）。単方向なら
conv の窓端だけで 5.1e-03 に収まるが、この形は双方向なので**成立しない**。記号長にした今、
pad する理由は 1 つも残っていない（丸める先の格子が無い）。

## 何をグラフに載せるか

`Crnn.forward` そのもの（conv 2 段 → 2 層 BiGRU → Linear）。出す値は **log_softmax 前の
ロジット**で、後処理（log_softmax → 遷移ペナルティ付き Viterbi → 短区間マージ → cons 吸収
→ `.lab`）も、特徴抽出（80 次元 log-mel + DSP 3 次元）もホスト側の責務。前者はロジットの
まま置くと golden の数値回帰の感度が落ちないため、後者はグラフに載せる理由が無いため。

batch は**静的 1**。動的軸は時間 `T` の 1 本だけ（`symbol_names=("T",)`）。

## 入力の約束（特徴抽出はグラフに載せない）

`features` の 83 次元の内訳（正本は上流 `training/src/vowel_detector/features.py`）:

- 0..79 — log-mel（16kHz / n_fft 512 / win 400 / hop 160 / n_mels 80）を**発話内 mean/std で
  z 化**したもの
- 80 — 有声性（自己相関のピーク比・`[0, 1]`）
- 81 — 発話ピーク比の log エネルギーを 10 で割ったもの（0 以下）
- 82 — 零交差率（`[0, 1]`）

{@link build_cases} の合成ケースはこの値域に収める（実測レンジは同関数の docstring）。

## 出力レイアウト

系列名は `vowel-detector-<チェックポイント名>`（既定の `crnn_epoch3.pt` なら
`vowel-detector-crnn-epoch3`）。**長さは綴りに入らない** — グラフが 1 本だからで、
系列を分けるのはチェックポイントの世代だけ:

    outputs/series/<系列名>/model.safetensors     重み・定数 + __metadata__
    outputs/series/<系列名>/io.<case>.safetensors 入力と torch CPU 期待出力

io のテンソルキー規約は tiny golden / DeBERTa / SigLIP2 と同じ
（`input.<グラフ入力名>` / `output.<位置>`）。
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file

from .patch import gru_forward

#: 実重みの親（手置きの入力素材 — docs/assets-layout.md）。
MODELS_ROOT = INPUTS_ROOT / "vowel-detector"

#: 既定のチェックポイント（`--ckpt` 未指定のとき）。
DEFAULT_CKPT = MODELS_ROOT / "crnn_epoch3.pt"

#: golden とトレース例の長さ（T10 = 10ms フレーム数 — 200 で 2.0 秒）。**グラフの受理集合では
#: ない**（記号軸なので任意長が 1 本のグラフで通る）。
DEFAULT_LENGTH = 200

#: 長さの刻み（出力の 20ms 格子 — モジュール docstring の「長さ軸」）。
LENGTH_MULTIPLE = 2

#: 最小の長さ。出力 2 フレーム = 逆方向 GRU が状態を 1 度は運ぶ最小形（1 フレームだと
#: 時間方向の再帰が消え、展開の誤りが値に出ない）。
MIN_LENGTH = 4

#: 記号 `T`（= 20ms フレーム数）の下限。`MIN_LENGTH` と同じ事実の記号側の綴りで、
#: 0 / 1 特殊化を避ける線でもある（convert の 2 点評価は下限 2 以上を要求する）。
SYM_MIN = MIN_LENGTH // LENGTH_MULTIPLE

#: 記号 `T` の上限 = **配布形が宣言する運用上限**（`karume/dist.py` の
#: `VOWEL_DETECTOR_MAX_FRAMES` が 10ms フレームへ直したものを配り、パイプラインが
#: 超過を fail loudly にする）。
#:
#: 30,000 × 20ms = **10 分**の音声。導出は最大中間テンソルの大きさ: conv 出力
#: `[1, 160, 2T]` f32 = 640 B / 10ms フレームなので、10 分で 38.4MiB。WebGPU の仕様既定
#: `maxStorageBufferBindingSize` 128MiB に対して 3.4 倍の余裕がある（karume はアダプタの
#: 実測値を要求するので実機ではさらに広い — `gpu/device.ts`）。
#:
#: MUST: 上げるときは中間テンソルの上限を測り直す。宣言だけ伸ばすと、超過は配布形の門ではなく
#: **利用者の手元の確保失敗**として出る。
SYM_MAX = 30_000

MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: グラフ入力の名前（= {@link Crnn.forward} の引数名）。export 後に突合してずれていたら
#: 止める — 位置で渡す以上、黙ってずれると golden の入力だけが入れ替わる。
INPUT_NAME = "features"

#: `.pt` の外側ラッパに期待する鍵（学習台本が `{"model": state_dict, "epoch": int}` で書く）。
#: 増減は上流の変更なので fail loudly。
WRAPPER_KEYS = ("model", "epoch")
#: state_dict を持つ鍵。
STATE_DICT_KEY = "model"

#: 畳み込み定数の格納キーの接頭辞（`karume.convert` が焼いた定数に付ける）。重み由来の
#: initializer と区別するために持つ。
CONST_PREFIX = "const."

#: 特徴の次元（log-mel 80 + DSP 3 — モジュール docstring の「入力の約束」）。
N_MELS = 80
DSP_DIM = 3
FEATURE_DIM = N_MELS + DSP_DIM

#: モデル形（上流 `crnn.py` の既定値そのまま）。
CONV_HIDDEN = 160
GRU_HIDDEN = 128
GRU_LAYERS = 2

#: 出力クラス（上流 `phonemes.LIPSYNC_CLASSES` と `assets/feature_config.json` の `classes`）。
CLASSES: tuple[str, ...] = ("a", "i", "u", "e", "o", "N", "pau", "cons")

#: 無音クラスの位置と母音クラスの本数（{@link _sanity} が確率質量を引く先）。
PAU_INDEX = CLASSES.index("pau")
VOWEL_COUNT = 5

#: 合成ケースの乱数（`noise` ケース）。グローバル seed に依存しない。
SEED = 20260813

#: {@link _sanity} が順序を見るケース名（無音らしい 1 本と有声らしい 1 本）。
SILENCE_CASE = "silence"
VOICED_CASE = "voiced"


class Crnn(nn.Module):
    """上流 `training/src/vowel_detector/crnn.py` の逐語の写し（モジュール docstring の
    「依存グループを足さない理由」）。

    conv で時間方向を 1/2 に畳み（10ms → 20ms）、2 層 BiGRU（オフライン前提なので双方向）を
    通して 8 クラスのロジットを出す。既定の寸法は上流と同じで、pytest が tiny な合成重みで
    同じ経路を回せるように**引数で受ける**（実重みは `load_state_dict(strict=True)` が
    形ごと検査する）。
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden: int = CONV_HIDDEN,
        gru_hidden: int = GRU_HIDDEN,
        gru_layers: int = GRU_LAYERS,
        classes: int = len(CLASSES),
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            hidden,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(gru_hidden * 2, classes)

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        """conv 2 段（10ms → 20ms 格子）。差し替え経路と参照経路が共有する前段。"""
        return self.conv(features.transpose(1, 2)).transpose(1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`[B, T10, feature_dim]` → `[B, T10//2, classes]` のロジット。

        GRU は `karume::gru_scan` の並びへ差し替えて通す（モジュール docstring の
        「差し替え層」）— これが export される経路。
        """
        return self.head(gru_forward(self.gru, self._encode(features)))

    def reference_forward(self, features: torch.Tensor) -> torch.Tensor:
        """`nn.GRU` をそのまま通す参照経路（export はしない）。

        {@link forward} との差はまさに差し替えぶんなので、両者の**ビット一致**が差し替えの
        正しさの主張そのものになる（{@link _write_io} の常設門と `--verify`）。
        """
        hidden, _ = self.gru(self._encode(features))
        return self.head(hidden)


def default_out_dir(ckpt: Path) -> Path:
    """生成物の既定の置き場（`outputs/series/vowel-detector-<ckpt 名>/`）。

    綴りへ焼くのは**チェックポイント名だけ** — 長さは記号軸なのでグラフは 1 本きりで、
    系列を分ける軸は epoch の世代しか残っていない（別の epoch を同じ席へ書くと先の重みが
    黙って上書きされる）。系列名は既存の流儀（小文字ハイフン）へ倒す。
    """
    return SERIES_ROOT / f"{MODELS_ROOT.name}-{ckpt.stem.lower().replace('_', '-')}"


def assert_length(length: int) -> None:
    """長さが刻みに乗っていることを見る（モジュール docstring の「長さ軸」）。"""
    if length < MIN_LENGTH or length % LENGTH_MULTIPLE:
        raise SystemExit(
            f"--length {length} は {MIN_LENGTH} 以上の {LENGTH_MULTIPLE} の倍数でない"
            "（出力は 20ms 格子 = 入力 2 フレームで 1 本）"
        )


def load_checkpoint(ckpt: Path) -> dict[str, torch.Tensor]:
    """`.pt` の外側ラッパを検査して state_dict を返す。

    MUST: `map_location="cpu"` — 上流の学習台本は CUDA で保存しているので、指定が無い環境
    では CUDA デバイスを掴もうとして落ちる。
    """
    if not ckpt.is_file():
        raise SystemExit(f"チェックポイントが見つからない: {ckpt}")
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or set(obj) != set(WRAPPER_KEYS):
        keys = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise SystemExit(
            f"{ckpt}: 外側ラッパの鍵が期待と違う（期待 {sorted(WRAPPER_KEYS)} / 実際 {keys}）"
        )
    state_dict = obj[STATE_DICT_KEY]
    non_tensor = [key for key, value in state_dict.items() if not torch.is_tensor(value)]
    if non_tensor:
        raise SystemExit(f"{ckpt}: state_dict にテンソルでない値がある: {sorted(non_tensor)}")
    return {key: value.detach().contiguous() for key, value in state_dict.items()}


def load_module(state_dict: Mapping[str, torch.Tensor]) -> Crnn:
    """写したモデル定義へ実重みを読み込む（`strict=True` — 写しのずれはここで落ちる）。"""
    module = Crnn()
    module.load_state_dict(dict(state_dict), strict=True)
    return module.eval()


def _features(
    length: int, mel: torch.Tensor, voicing: float, log_energy: float, zcr: float
) -> torch.Tensor:
    """log-mel 面と定数 DSP 3 次元を繋いで `[1, T10, 83]` にする。"""
    dsp = torch.tensor([[voicing, log_energy, zcr]]).expand(length, DSP_DIM)
    return torch.cat([mel, dsp], dim=1).unsqueeze(0)


def build_cases(length: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """golden 4 ケースの `(名前, features)`（合成の特徴 `[1, T10, 83]`）。

    実音声を使わないのは、特徴抽出（log-mel + DSP）がグラフの外だから — この台本の主張は
    「同じ特徴を入れたら同じロジットが出る」で、音声の decode は含まない。

    値は実発話（1.1 秒）の実測レンジに収める: log-mel の z 値 −1.04〜3.72 / 有声性
    0.08〜0.90 / log エネルギー比 −0.73〜0 / 零交差率 0.04〜0.85。

    {@link SILENCE_CASE} と {@link VOICED_CASE} は {@link _sanity} の順序が恒真にならない
    ための対で、`noise` / `ramp` は数値回帰の検出用（値域の端と勾配を踏む）。
    """
    generator = torch.Generator().manual_seed(SEED)
    # 無音: log-mel が一様に下限、有声性も零交差率も低く、エネルギーは発話ピークの −7.2 桁。
    silence = torch.full((length, N_MELS), -1.0)
    # 有声: 低域 20 本が強く、中域 30 本で下がり、高域 30 本は下限（母音のスペクトル傾斜）。
    voiced = torch.cat(
        [
            torch.full((20,), 2.0),
            torch.linspace(1.5, -1.0, 30),
            torch.full((30,), -1.0),
        ]
    ).expand(length, N_MELS)
    noise = torch.rand((length, N_MELS), generator=generator) * 4.0 - 1.0
    ramp = torch.linspace(-1.0, 3.0, N_MELS).reshape(1, N_MELS) + torch.linspace(
        -1.0, 1.0, length
    ).reshape(length, 1)
    return (
        (SILENCE_CASE, _features(length, silence, 0.05, -0.72, 0.10)),
        (VOICED_CASE, _features(length, voiced, 0.85, 0.0, 0.05)),
        ("noise", _features(length, noise, 0.5, -0.3, 0.5)),
        ("ramp", _features(length, ramp, 0.3, -0.5, 0.3)),
    )


def _write_io(
    module: Crnn,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    MUST: 期待値は**参照経路**（`nn.GRU` そのもの — {@link Crnn.reference_forward}）から採り、
    export される差し替え経路とのビット一致を毎ケース見る。差し替えの誤りが期待値の側へ
    そのまま乗ると、E2E は「エクスポータとランタイムが一致して両方間違っている」状態で緑に
    なる（`karume/custom_ops.py` の docstring と同じ罠）。

    戻り値の 2 本目は sanity 記録用の期待出力（`[1, T10//2, 8]` の形のまま渡す）。
    """
    if len(graph.outputs) != 1:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（ロジットは 1 本）")
    written: list[str] = []
    logits: dict[str, torch.Tensor] = {}
    for name, features in cases:
        with torch.no_grad():
            output = module.reference_forward(features)
            patched = module(features)
        if not torch.equal(patched, output):
            raise AssertionError(
                f"{name}: gru_scan への差し替えが nn.GRU とビット一致しない"
                f"（maxdiff {float((patched - output).abs().max())}）"
            )
        # MUST: io も IR の意味論 dtype の実表現へ落とす（ADR 0009 の境界正規化）。ランタイムが
        # 受け取る形と揃っていないと Deno 側 E2E が golden を読めない。
        tensors = {
            f"{INPUT_PREFIX}{INPUT_NAME}": normalize_boundary_tensor(
                features, f"{name} の入力 '{INPUT_NAME}'"
            ),
            f"{OUTPUT_PREFIX}0": normalize_boundary_tensor(
                output.detach().contiguous(), f"{name} の出力 0"
            ),
        }
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        logits[name] = output.detach()
    return written, logits


def _sanity(logits: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """無音らしい入力と有声らしい入力の**順序**を見る（閾値は置かない）。

    MUST: 恒真な sanity にしない。ロジットの値域検査も「同じ入力どうしが一致する」も何も
    検証しない。①無音ケースの平均 P(pau) が全ケース中で最大 ②有声ケースの平均母音質量
    （a/i/u/e/o の和）が全ケース中で最大、の 2 つの順序で見れば、重みのレイアウト取り違え
    （時間軸と特徴軸の転置・GRU のゲート順の誤り）も、入力に依存しなくなった出力も落ちる。

    ケースどうしが**互いに違う**ことも見る（同じ出力が 4 本並ぶのは入力が届いていない形）。
    """
    names = sorted(logits)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if torch.equal(logits[name], logits[other]):
                raise AssertionError(f"{name} と {other} の出力が同一（入力が効いていない）")

    probabilities = {name: tensor.softmax(dim=-1) for name, tensor in logits.items()}
    pause = {name: float(tensor[..., PAU_INDEX].mean()) for name, tensor in probabilities.items()}
    vowel = {
        name: float(tensor[..., :VOWEL_COUNT].sum(dim=-1).mean())
        for name, tensor in probabilities.items()
    }
    _assert_largest(SILENCE_CASE, pause, "無音ケースの平均 P(pau) が最大でない")
    _assert_largest(VOICED_CASE, vowel, "有声ケースの平均母音質量が最大でない")
    return {
        "pau_mean": {name: round(value, 4) for name, value in pause.items()},
        "vowel_mass_mean": {name: round(value, 4) for name, value in vowel.items()},
    }


def _assert_largest(expected: str, values: Mapping[str, float], claim: str) -> None:
    """`expected` が `values` の最大でなければ `claim` を掲げて落とす（順序だけを見る）。"""
    winner = max(values, key=lambda name: values[name])
    if winner != expected:
        raise AssertionError(
            f"{claim}: {expected}={values[expected]:.4f} < {winner}={values[winner]:.4f}"
        )


def assert_checkpoint_bytes(path: Path, state_dict: Mapping[str, torch.Tensor]) -> int:
    """emit した initializer が上流 `.pt` の state_dict と**バイト一致**することを見る。

    重みの変換は「読んで書くだけ」なので、値が変われば変換が壊れている。dtype・shape だけ
    でなく生バイト列で突き合わせる（NaN のビット列や −0.0 まで含めて「値が変わっていない」を
    主張するため）。読み直しは**別実装のリーダ**（`safetensors.safe_open`）で行う。

    重み由来でない initializer（畳み込みで焼かれた定数）は `const.` 接頭辞を持つので、
    それ以外の余剰キーは「重みが黙って書き換えられて別名で入った」形として落とす。
    一致した本数を返す。
    """
    with safe_open(str(path), framework="pt") as handle:
        keys = set(handle.keys())
        missing = sorted(set(state_dict) - keys)
        if missing:
            raise AssertionError(f"{path}: state_dict の鍵が initializer に無い: {missing}")
        extra = sorted(key for key in keys - set(state_dict) if not key.startswith(CONST_PREFIX))
        if extra:
            raise AssertionError(f"{path}: 重み由来でない initializer がある: {extra}")
        for name in sorted(state_dict):
            source = state_dict[name]
            stored = handle.get_tensor(name)
            if stored.dtype != source.dtype or tuple(stored.shape) != tuple(source.shape):
                raise AssertionError(
                    f"テンソル '{name}': dtype / shape 不一致"
                    f"（元 {source.dtype} {tuple(source.shape)} /"
                    f" 読み直し {stored.dtype} {tuple(stored.shape)}）"
                )
            if stored.numpy().tobytes() != source.numpy().tobytes():
                raise AssertionError(f"テンソル '{name}': バイト列が一致しない")
    return len(state_dict)


def export_series(ckpt: Path, out_dir: Path, length: int) -> dict[str, Any]:
    """IR コンテナと golden io を書き、要約を返す。"""
    assert_length(length)
    state_dict = load_checkpoint(ckpt)
    module = load_module(state_dict)
    cases = build_cases(length)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, example = cases[0]
    started = time.monotonic()
    graph = export_to_file(
        module,
        (example,),
        out_dir / MODEL_FILE,
        # MUST: 記号は出力の 20ms 格子側に置く（`2*Dim("T")` — モジュール docstring の
        # 「長さ軸」）。素の `Dim("T")` だと conv の出力が床除算になり次元言語に載らない。
        dynamic_shapes={INPUT_NAME: {1: 2 * Dim("T", min=SYM_MIN, max=SYM_MAX)}},
        symbol_names=("T",),
    )
    elapsed = time.monotonic() - started
    declared = tuple(item.name for item in graph.inputs)
    if declared != (INPUT_NAME,):
        raise AssertionError(f"グラフ入力の並びが {declared} で、期待の {(INPUT_NAME,)} と違う")
    matched = assert_checkpoint_bytes(out_dir / MODEL_FILE, state_dict)
    written, logits = _write_io(module, graph, cases, out_dir)
    return {
        "dir": str(out_dir),
        "length": length,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": (out_dir / MODEL_FILE).stat().st_size,
        "export_seconds": round(elapsed, 2),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "declared_input_shape": [str(dim) for dim in graph.inputs[0].shape],
        "input_shape": list(example.shape),
        "output_shape": list(logits[cases[0][0]].shape),
        "symbol_range": [SYM_MIN, SYM_MAX],
        "checkpoint_tensors_byte_identical": matched,
        "sanity": _sanity(logits),
    }


def _diff_entry(
    stage: str,
    claim: str,
    got: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """1 段分の同値レポート（ケース名 → maxdiff と、全ケースのビット一致）。

    `bit_exact` は「差 0」より強い主張（`0.0 == -0.0` は差 0 だがビットは違う）。
    """
    return {
        "stage": stage,
        "claim": claim,
        "maxdiff": {
            name: float((got[name] - expected[name]).abs().max()) for name in sorted(expected)
        },
        "bit_exact": all(torch.equal(got[name], expected[name]) for name in expected),
    }


#: `--verify` が踏む長さ（`--length` に加えて**複数長**を見る）。差し替えの誤りの中には
#: 長さに依存して現れるもの（逆方向の走査境界・層間の連結順）があり、1 長だけでは踏めない。
VERIFY_LENGTHS: tuple[int, ...] = (MIN_LENGTH, 6, 18, 137 * 2)


def verify_patch(ckpt: Path, length: int) -> list[dict[str, Any]]:
    """`gru_scan` への差し替えが `nn.GRU` と同値であることを実測する（`--verify`）。

    他の台本の `--verify`（差し替え前後の eager 同値）と同じ主張で、相手は
    {@link Crnn.reference_forward} = 上流そのままの `nn.GRU` 経路。ここが崩れると
    「グラフは焼けるが数値が別物」になるので、**bit_exact が主張の中身**（差 0 より強い —
    {@link _diff_entry}）。

    NOTE: emit しないのは他の台本と同じ形に揃えるためで、こちらにプロセス汚染の危険は無い
    （クラス属性の差し替えを一切しない — 差し替えは `Crnn.forward` の中だけ）。
    """
    assert_length(length)
    module = load_module(load_checkpoint(ckpt))
    entries: list[dict[str, Any]] = []
    for probe in sorted({length, *VERIFY_LENGTHS}):
        cases = build_cases(probe)
        with torch.no_grad():
            reference = {name: module.reference_forward(features) for name, features in cases}
            got = {name: module(features) for name, features in cases}
        entry = _diff_entry(f"gru-scan-t{probe}", "bit-exact", got, reference)
        if not entry["bit_exact"]:
            raise AssertionError(
                f"T10={probe}: gru_scan への差し替えが nn.GRU とビット同一でない:"
                f" maxdiff={entry['maxdiff']}"
            )
        entries.append(entry)
    return entries


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--length",
        type=int,
        default=DEFAULT_LENGTH,
        help=f"golden ケースの 10ms フレーム数 T10（{LENGTH_MULTIPLE} の倍数・既定"
        f" {DEFAULT_LENGTH}）。**グラフの受理集合ではない** — 時間軸は記号なので、"
        "焼かれる 1 本のグラフが任意長を受ける",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は outputs/series/vowel-detector-<ckpt 名>/）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="gru_scan への差し替えと nn.GRU の同値を実測する（emit はしない）",
    )
    args = parser.parse_args(argv)
    if args.verify:
        for entry in verify_patch(args.ckpt, args.length):
            print(
                f"{entry['stage']} ({entry['claim']}): bit_exact={entry['bit_exact']}"
                f" maxdiff={entry['maxdiff']}"
            )
        return
    out_dir = args.out if args.out is not None else default_out_dir(args.ckpt)
    summary = export_series(args.ckpt, out_dir, args.length)
    print(json.dumps({"ckpt": str(args.ckpt), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
