"""EmbeddingGemma-300m の**量子化方式**を torch CPU で横並びに測る（方式スクリーニング）。

`anima/measure_quant.py` / `irodori/measure_quant.py` と同じ思想の品質軸 — **GPU コードは
0 行**で、ここで測るのは量子化そのものの質だけ（ADR 0006 の fake-quant 方法論では E2E は
実装誤差しか測らないので、品質は別軸で測る必要がある）。答えたい問いは 2 つ:

    ① **4bit の格子の張り方で埋め込みの質は変わるか** — RTN(i4) / FP4 / NF4 / MXFP4 /
       k-means codebook 3 粒度を、同じ group（g=32 固定）・同じ対象で横並びにする。
    ② **語彙表を 4bit へ落とせるか** — EG は `embed_tokens` が 201M / 307M（配布サイズの
       約 66%）を占めるので、量子化の旨みはここに集中している。対象を「linear 限定」と
       「linear + embedding」の 2 形で走らせ、差を語彙表の寄与として読む。

NOTE: 対象 `linear+embedding` は**測定専用の形**。i4 の実行経路は消費 op が linear の重み
スロットに限られる（ADR 0069 決定 5 / `docs/ir-v1.md` の `i4` 格納形）ので、embedding を
4bit で「格納」する道はいま無い。ここで測るのは「その席が開いたときに何が起きるか」で、
出荷形の主張ではない（`quant_methods` の測定専用方式も同じ立場）。

## 構成（8 = baseline + 方式 7 × 対象 2 のうち baseline は対象に依らないので 1 本）

| 方式             | 丸め                                                     |
| ---------------- | -------------------------------------------------------- |
| `f32`            | 丸めなし（**基準** — 全ての cosine はこの埋め込みとの比） |
| `rtn-i4-g32`     | `karume.quantize.fake_quant_int4`（出荷形の格納そのもの） |
| `fp4`            | FP4 (e2m1) 固定表 × group absmax                          |
| `nf4`            | NF4 固定表 × group absmax                                 |
| `mxfp4`          | FP4 表 × group の **2 のべき** scale（OCP MX）            |
| `kmeans:*`       | 16 centroid の codebook（per_tensor / per_channel / shared）|

group は **g=32 固定**（方式間の比較を格子の張り方 1 本に絞るため — group 長の軸は ADR 0069
Phase 0 の sweep が既に持っている）。

MUST: **方式を積み重ねない** — 全ての丸めは in-place なので、構成ごとに pristine（素の f32
重み）へ戻してから当てる（{@link restore}）。戻さずに当てると測っているのが「NF4 の上の
MXFP4」になり、方式間の比較そのものが意味を失う。模型のロードは 1 回きり。

## 測定列

- **ケース別 cosine**（`export.py` の golden 5 ケース）— 基準 f32 の埋め込みとの cosine の
  min と mean。埋め込みは L2 正規化済みなので内積そのものだが、正規化つきで採る。
- **意味順序の保持** — 近い対（{@link export.NEAR_PAIR}）の cosine が遠い対
  （{@link export.FAR_PAIR}）を上回るか。`export._sanity` と同じ対で、「数値は動いているが
  埋め込みとして壊れている」を基準比 cosine とは独立に捕まえる列（cosine が 0.99 でも
  順序が壊れれば検索は壊れる）。
- **ペア間 cosine 行列のドリフト** — 5 ケース総当たり 10 対の cosine を f32 と比べた
  最大絶対差。埋め込みの使い道は「ベクトル同士の比較」なので、基準ベクトルとの近さより
  こちらが実運用に近い（全ベクトルが同じ向きへ回っても検索順位は動かない）。
- **サイズ試算** — 方式ごとの実効 bpw と対象集合の投影 MiB（式は {@link Method.formula}）。

## 出力

    uv run --with 'transformers==5.14.1' python -m embeddinggemma.measure_quant --out <dir>

`<out>/report.json` に全数値、stdout へ Markdown 表 2 枚（品質 / サイズ試算）。進捗は
stderr へ即 flush する（数十分級の背景実行で追うため）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from _shared.paths import OUTPUTS_ROOT
from karume.quant_methods import (
    DEFAULT_CODEBOOK_LEVELS,
    fake_quant_fp4,
    fake_quant_kmeans,
    fake_quant_mxfp4,
    fake_quant_nf4,
)
from karume.quantize import channel_rows, fake_quant_int4, iter_quant_targets

from . import export as eg

#: 生成物の既定の置き場（デモ・測定の生成物は系列と分ける — `irodori/measure_quant.py` と同じ）。
DEFAULT_OUT = OUTPUTS_ROOT / "demo" / "embeddinggemma-quant-screen"

REPORT_FILE = "report.json"

#: 基準構成の綴り（丸めなし — 全ての比較の分母）。
BASE_CONFIG = "f32"

#: 全方式で固定する group 長（方式間の比較を「格子の張り方」1 本に絞るための固定 —
#: モジュール docstring）。group 長そのものの軸は ADR 0069 Phase 0 の sweep が持つ。
GROUP_SIZE = 32

# ---- サイズ試算の bit 幅（格納規則の逐語）------------------------------------
#
# 出典は `docs/ir-v1.md` の `i4` 格納形（scale は **F32**・group ごと 1 個・group_size は 2 冪
# かつ 16 以上）と OCP Microscaling Formats v1.0（MX の共有 scale は E8M0 = 指数 1 バイト）。
# k-means は格納形を持たない測定専用方式なので、**表のコストを込みで**素直に数える。

#: 4bit 格子のペイロード（全方式共通 — 比較しているのは「格子の張り方」であって bit 数ではない）。
PAYLOAD_BITS = 4.0
#: group scale の bit 幅（`i4` の格納は F32 の group scale が MUST — `docs/ir-v1.md`）。
F32_SCALE_BITS = 32.0
#: MXFP4 の共有 scale は E8M0（指数 1 バイト）。
MX_SCALE_BITS = 8.0
#: codebook 1 エントリの bit 幅（centroid を F32 で持つ）。
CODEBOOK_ENTRY_BITS = 32.0

BITS_PER_BYTE = 8
MIB = 1024 * 1024
#: 基準（丸めなし）の 1 要素あたり bit。
F32_BITS = 32.0


@dataclass(frozen=True)
class TargetStats:
    """対象テンソル集合の素性（サイズ試算の入力）。

    `groups` は g=32 の group 数の総和、`channels` は per-channel 表のコストが比例する行数
    （= {@link karume.quantize.channel_rows} の行 = 出力チャネル）の総和。
    """

    modules: int
    elements: int
    groups: int
    channels: int


@dataclass(frozen=True)
class Target:
    """量子化の対象範囲（`op_types` で切り替える — 対象選択の正本は core 側）。"""

    name: str
    op_types: tuple[type[nn.Module], ...]


#: transformer の linear + SentenceTransformer の Dense 2 段（出荷形の i4 適格と同じ集合）。
TARGET_LINEAR = Target("linear", (nn.Linear,))
#: 上に `embed_tokens` を足した形（EG の配布サイズの支配項 — モジュール docstring の NOTE）。
TARGET_WITH_EMBEDDING = Target("linear+embedding", (nn.Linear, nn.Embedding))
TARGETS: tuple[Target, ...] = (TARGET_LINEAR, TARGET_WITH_EMBEDDING)

#: pristine 退避と復元の対象（**最も広い形**で採る — 対象の狭い構成でも同じ退避から戻す）。
WIDEST_OP_TYPES = TARGET_WITH_EMBEDDING.op_types


def _group_scaled_bits(scale_bits: float) -> Callable[[TargetStats], float]:
    """group scale を持つ方式の総 bit（`4·N + scale_bits·G`）。"""

    def bits(stats: TargetStats) -> float:
        return PAYLOAD_BITS * stats.elements + scale_bits * stats.groups

    return bits


def _codebook_bits(
    tables: Callable[[TargetStats], int], scale_bits: float
) -> Callable[[TargetStats], float]:
    """codebook 方式の総 bit（`4·N + scale_bits·G + 32·16·表の枚数`）。"""

    def bits(stats: TargetStats) -> float:
        table_bits = CODEBOOK_ENTRY_BITS * DEFAULT_CODEBOOK_LEVELS * tables(stats)
        return PAYLOAD_BITS * stats.elements + scale_bits * stats.groups + table_bits

    return bits


@dataclass(frozen=True)
class Method:
    """1 方式のレシピ。

    `apply` は**素の f32 重みへ**丸めを当てて計数の 1 行要約を返す（積み重ねの禁止は
    呼び出し側の {@link restore} が担保する）。`bits` はサイズ試算で、`formula` はその式を
    読み手が検算できる形の逐語。
    """

    name: str
    apply: Callable[[nn.Module, tuple[type[nn.Module], ...]], str]
    bits: Callable[[TargetStats], float]
    formula: str


METHODS: tuple[Method, ...] = (
    Method(
        "rtn-i4-g32",
        lambda model, op_types: fake_quant_int4(model, GROUP_SIZE, op_types=op_types).describe(),
        _group_scaled_bits(F32_SCALE_BITS),
        "4·N + 32·G（G = N/32 の group ごとに F32 scale 1 個）",
    ),
    Method(
        "fp4",
        lambda model, op_types: fake_quant_fp4(model, GROUP_SIZE, op_types=op_types).describe(),
        _group_scaled_bits(F32_SCALE_BITS),
        "4·N + 32·G（i4 と同じ group absmax scale・格子だけが指数的）",
    ),
    Method(
        "nf4",
        lambda model, op_types: fake_quant_nf4(model, GROUP_SIZE, op_types=op_types).describe(),
        _group_scaled_bits(F32_SCALE_BITS),
        "4·N + 32·G（同上・格子が正規分布の分位点）",
    ),
    Method(
        "mxfp4",
        lambda model, op_types: fake_quant_mxfp4(model, GROUP_SIZE, op_types=op_types).describe(),
        _group_scaled_bits(MX_SCALE_BITS),
        "4·N + 8·G（共有 scale が E8M0 の 2 のべき — OCP MX v1.0）",
    ),
    Method(
        "kmeans:per_tensor",
        lambda model, op_types: fake_quant_kmeans(
            model, "per_tensor", GROUP_SIZE, op_types=op_types
        ).describe(),
        _codebook_bits(lambda stats: stats.modules, 0.0),
        "4·N + 32·16·(テンソル数)（scale 無し・表は層ごと 1 枚）",
    ),
    Method(
        "kmeans:per_channel",
        lambda model, op_types: fake_quant_kmeans(
            model, "per_channel", GROUP_SIZE, op_types=op_types
        ).describe(),
        _codebook_bits(lambda stats: stats.channels, 0.0),
        "4·N + 32·16·(出力チャネル数)（scale 無し・表は行ごと 1 枚）",
    ),
    Method(
        "kmeans:shared",
        lambda model, op_types: fake_quant_kmeans(
            model, "shared", GROUP_SIZE, op_types=op_types
        ).describe(),
        _codebook_bits(lambda _stats: 1, F32_SCALE_BITS),
        "4·N + 32·G + 32·16（表はモデル全体で 1 枚・group absmax scale つき）",
    ),
)

METHOD_NAMES = tuple(method.name for method in METHODS)

#: ケース名の並び（golden の宣言順 — ペア行列の行列順もこれ）。
CASE_NAMES: tuple[str, ...] = tuple(name for name, _prompt, _body in eg.GOLDEN_CASES)


def config_name(method: Method, target: Target) -> str:
    return f"{method.name}/{target.name}"


# ---- 指標 -------------------------------------------------------------------


def cosine(value: torch.Tensor, reference: torch.Tensor) -> float:
    """`⟨v,r⟩ / (‖v‖·‖r‖)` を f64 で採る。

    埋め込みはグラフの最終段で L2 正規化されているので内積そのものになるはずだが、
    正規化つきで採る — 丸めがノルムを動かした場合に「内積が 1 を超える / 下回る」形で
    ノルムの崩れが cosine 側へ混ざるのを避ける（ノルムの崩れは別の量）。
    """
    x = value.reshape(-1).to(torch.float64)
    y = reference.reshape(-1).to(torch.float64)
    denom = float(x.norm() * y.norm())
    if denom == 0.0:
        raise AssertionError("cosine の分母が 0（埋め込みが全ゼロ — 丸め以前に経路が壊れている）")
    return float(torch.dot(x, y) / denom)


def pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}×{pair[1]}"


def pair_cosines(vectors: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """5 ケース総当たり（上三角 10 対）の cosine。"""
    return {
        pair_label((left, right)): cosine(vectors[left], vectors[right])
        for index, left in enumerate(CASE_NAMES)
        for right in CASE_NAMES[index + 1 :]
    }


def l2_norms(vectors: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(vector.to(torch.float64).norm()) for name, vector in vectors.items()}


def measure(
    vectors: Mapping[str, torch.Tensor], base: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """1 構成ぶんの測定列（基準 f32 の埋め込みとの比較）。"""
    per_case = {name: cosine(vectors[name], base[name]) for name in CASE_NAMES}
    pairs = pair_cosines(vectors)
    base_pairs = pair_cosines(base)
    drift = {label: pairs[label] - base_pairs[label] for label in pairs}
    worst = max(drift, key=lambda label: abs(drift[label]))
    near, far = pairs[pair_label(eg.NEAR_PAIR)], pairs[pair_label(eg.FAR_PAIR)]
    base_near = base_pairs[pair_label(eg.NEAR_PAIR)]
    base_far = base_pairs[pair_label(eg.FAR_PAIR)]
    values = list(per_case.values())
    return {
        "caseCosine": per_case,
        "caseCosineMin": min(values),
        "caseCosineMean": sum(values) / len(values),
        "l2Norms": l2_norms(vectors),
        "order": {
            "near": pair_label(eg.NEAR_PAIR),
            "far": pair_label(eg.FAR_PAIR),
            "nearCosine": near,
            "farCosine": far,
            "margin": near - far,
            "baseMargin": base_near - base_far,
            "holds": near > far,
        },
        "pairs": pairs,
        "pairDrift": drift,
        "pairDriftMaxAbs": abs(drift[worst]),
        "pairDriftWorst": worst,
    }


# ---- 対象集合とサイズ試算 ---------------------------------------------------


def target_stats(model: nn.Module, target: Target) -> TargetStats:
    """対象集合の本数・要素数・group 数・チャネル数（サイズ試算の入力）。

    group 数は {@link karume.quantize.channel_rows} の平坦形から数える — 丸めが group を
    割る軸（各 op の「in 軸」）と同じ軸で数えないと、試算だけが別の形の数になる。
    量子化軸が `GROUP_SIZE` で割り切れない対象は fail loudly（格納できない形の bpw を
    出さない — ADR 0069 決定 2）。
    """
    modules = elements = groups = channels = 0
    for fqn, weight, axis in iter_quant_targets(model, target.op_types):
        rows = channel_rows(weight, axis)
        span = int(rows.shape[1])
        if span % GROUP_SIZE:
            raise AssertionError(
                f"'{fqn}': 量子化軸 {span} が group {GROUP_SIZE} で割り切れない"
                "（ADR 0069 決定 2 — 端数 group を作った形の試算は出さない）"
            )
        modules += 1
        elements += int(weight.numel())
        groups += int(rows.shape[0]) * (span // GROUP_SIZE)
        channels += int(rows.shape[0])
    if not modules:
        raise AssertionError(f"対象 '{target.name}' に量子化できる重みが 1 本も無い")
    return TargetStats(modules=modules, elements=elements, groups=groups, channels=channels)


def size_projection(method: Method, stats: TargetStats) -> dict[str, Any]:
    """方式 × 対象集合の実効 bpw と投影 MiB（対象テンソルだけの合計 — 模型全体ではない）。"""
    bits = method.bits(stats)
    baseline_bits = F32_BITS * stats.elements
    return {
        "elements": stats.elements,
        "modules": stats.modules,
        "groups": stats.groups,
        "channels": stats.channels,
        "bitsPerWeight": bits / stats.elements,
        "projectedMiB": bits / BITS_PER_BYTE / MIB,
        "f32MiB": baseline_bits / BITS_PER_BYTE / MIB,
        "ratio": bits / baseline_bits,
        "formula": method.formula,
    }


# ---- 実行 -------------------------------------------------------------------


def embed_cases(
    wrapper: nn.Module, cases: Sequence[tuple[str, torch.Tensor, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """golden 5 ケースの埋め込み（`[H]` へ平坦化した clone）。"""
    vectors: dict[str, torch.Tensor] = {}
    for name, ids, pool_mask in cases:
        with torch.no_grad():
            vectors[name] = wrapper(ids, pool_mask).detach().reshape(-1).clone()
    return vectors


def stacked(vectors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """ビット等値の判定用に 5 ケースを 1 本へ積む（順序は {@link CASE_NAMES} 固定）。"""
    return torch.stack([vectors[name] for name in CASE_NAMES])


def restore(weights: Mapping[str, torch.Tensor], pristine: Mapping[str, torch.Tensor]) -> None:
    """全対象を pristine の f32 重みへ戻す（**方式を積み重ねない** MUST の実体）。"""
    with torch.no_grad():
        for fqn, weight in weights.items():
            weight.copy_(pristine[fqn])


def select_methods(only: Sequence[str]) -> tuple[Method, ...]:
    """`--only` で選んだ方式を宣言順で返す（基準 f32 は常に走る — 比較の分母）。"""
    if not only:
        return METHODS
    chosen = set(only)
    return tuple(method for method in METHODS if method.name in chosen)


def run(model_dir: Path, methods: Sequence[Method]) -> dict[str, Any]:
    """模型を 1 回だけ読み、構成を順に当てて測る。"""
    cases = eg.build_cases(model_dir, eg.SYM_MAX)
    lengths = " ".join(f"{name}(T={int(ids.shape[1])})" for name, ids, _ in cases)
    print(f"[cases] {len(cases)} ケース: {lengths}", file=sys.stderr, flush=True)

    wrapper = eg.load_wrapper(model_dir)
    stats = {target.name: target_stats(wrapper, target) for target in TARGETS}
    for target in TARGETS:
        entry = stats[target.name]
        print(
            f"[target] {target.name}: {entry.modules} 本 / {entry.elements:,} 要素"
            f" / group {entry.groups:,} / チャネル {entry.channels:,}",
            file=sys.stderr,
            flush=True,
        )

    weights = {fqn: weight for fqn, weight, _axis in iter_quant_targets(wrapper, WIDEST_OP_TYPES)}
    pristine = {fqn: weight.detach().clone() for fqn, weight in weights.items()}

    base_vectors = embed_cases(wrapper, cases)
    # 基準そのものの門 — 単位ノルムと意味順序（`export._sanity` の逐語）。ここが割れたら
    # 以後の全数値の意味が消えるので、方式を 1 つも当てる前に落とす。
    base_sanity = eg._sanity(base_vectors)
    print(f"[{BASE_CONFIG}] sanity {json.dumps(base_sanity, ensure_ascii=False)}", file=sys.stderr)

    configs: dict[str, dict[str, Any]] = {
        BASE_CONFIG: {
            "method": BASE_CONFIG,
            "target": "—",
            "quantReport": "丸めなし（基準）",
            **measure(base_vectors, base_vectors),
        }
    }
    for method in methods:
        for target in TARGETS:
            name = config_name(method, target)
            started = time.perf_counter()
            restore(weights, pristine)
            report = method.apply(wrapper, target.op_types)
            vectors = embed_cases(wrapper, cases)
            entry: dict[str, Any] = {
                "method": method.name,
                "target": target.name,
                "quantReport": report,
                "moved": not torch.equal(stacked(vectors), stacked(base_vectors)),
                "size": size_projection(method, stats[target.name]),
                **measure(vectors, base_vectors),
                "elapsed": round(time.perf_counter() - started, 1),
            }
            configs[name] = entry
            print(
                f"[{name}] {report} cos min {entry['caseCosineMin']:.6f}"
                f" 順序 {'保持' if entry['order']['holds'] else '崩壊'}"
                f" ドリフト {entry['pairDriftMaxAbs']:.4e}"
                f" ({entry['elapsed']:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
    restore(weights, pristine)
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/embeddinggemma/measure_quant.py",
        "torch": torch.__version__,
        "modelDir": str(model_dir),
        "groupSize": GROUP_SIZE,
        "codebookLevels": DEFAULT_CODEBOOK_LEVELS,
        "cases": {name: int(ids.shape[1]) for name, ids, _ in cases},
        "baseSanity": base_sanity,
        "targets": {
            target.name: {
                "opTypes": [cls.__name__ for cls in target.op_types],
                **vars(stats[target.name]),
            }
            for target in TARGETS
        },
        "configs": configs,
        "gates": run_gates(configs, methods),
    }


def run_gates(
    configs: Mapping[str, Mapping[str, Any]], methods: Sequence[Method]
) -> dict[str, Any]:
    """恒真化の遮断（丸めの素通りと `op_types` の効き目）。

    ① 各構成の埋め込みが基準と**ビット一致しない** — 一致するのは丸めが 1 本も当たって
       いない場合だけで、しかも品質は「完璧」側に出るので黙ると誤読しかされない。
    ② `linear+embedding` が `linear` より**ちょうど embedding のぶんだけ多く**丸める
       （本数の差 = 1 本）。`op_types` が効いていなければここが 0 になり、「語彙表も測った」
       と読める表が語彙表を素通りしたまま出る。
    ③ ② の 2 形の埋め込みが**互いに違う** — 本数だけ増えて値が同じなら、embedding の丸めが
       出力へ届いていない（対象は増えたが経路が違う、を捕まえる）。
    """
    failures: list[str] = []
    for name, entry in configs.items():
        if name != BASE_CONFIG and not entry["moved"]:
            failures.append(f"{name}: 埋め込みが基準とビット一致（丸めが素通りしている）")
    embedding_targets: list[str] = []
    for method in methods:
        linear = configs[config_name(method, TARGET_LINEAR)]
        widened = configs[config_name(method, TARGET_WITH_EMBEDDING)]
        delta = widened["size"]["modules"] - linear["size"]["modules"]
        embedding_targets.append(f"{method.name}: +{delta} 本")
        if delta != 1:
            failures.append(
                f"{method.name}: linear+embedding が linear より {delta} 本多い"
                "（embed_tokens 1 本ぶんのはず — op_types の切り替えが効いていない）"
            )
        if widened["caseCosine"] == linear["caseCosine"]:
            failures.append(
                f"{method.name}: linear+embedding と linear の埋め込みが同一"
                "（語彙表の丸めが出力へ届いていない）"
            )
    return {
        "moved": {name: entry["moved"] for name, entry in configs.items() if name != BASE_CONFIG},
        "embeddingWidening": embedding_targets,
        "failures": failures,
    }


# ---- 表 ---------------------------------------------------------------------


def _markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def quality_table(report: Mapping[str, Any]) -> str:
    """方式 × 対象 × 測定列（この台本の主成果）。"""
    header = [
        "方式",
        "対象",
        "本数",
        "cos min",
        "cos mean",
        "near",
        "far",
        "意味順序",
        "ペア最大ドリフト",
    ]
    rows: list[list[str]] = []
    for name, entry in report["configs"].items():
        order = entry["order"]
        modules = entry["size"]["modules"] if "size" in entry else "—"
        rows.append(
            [
                name if name == BASE_CONFIG else entry["method"],
                entry["target"],
                str(modules),
                f"{entry['caseCosineMin']:.6f}",
                f"{entry['caseCosineMean']:.6f}",
                f"{order['nearCosine']:.4f}",
                f"{order['farCosine']:.4f}",
                "保持" if order["holds"] else "**崩壊**",
                f"{entry['pairDriftMaxAbs']:.4e}",
            ]
        )
    return _markdown(header, rows)


def size_table(report: Mapping[str, Any]) -> str:
    """方式 × 対象のサイズ試算（対象テンソル集合だけの合計）。"""
    header = ["方式", "対象", "要素数", "bpw", "投影 MiB", "f32 MiB", "比", "式"]
    rows = [
        [
            entry["method"],
            entry["target"],
            f"{entry['size']['elements']:,}",
            f"{entry['size']['bitsPerWeight']:.3f}",
            f"{entry['size']['projectedMiB']:.1f}",
            f"{entry['size']['f32MiB']:.1f}",
            f"{entry['size']['ratio'] * 100:.1f}%",
            entry["size"]["formula"],
        ]
        for entry in report["configs"].values()
        if "size" in entry
    ]
    return _markdown(header, rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=eg.DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=METHOD_NAMES,
        help="この方式だけ走らせる（複数可・部分再実行用）。基準 f32 は常に走る。"
        f" {REPORT_FILE} には走らせた構成だけが載る。",
    )
    args = parser.parse_args(argv)
    report = run(args.model_dir, select_methods(args.only))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("品質（基準 f32 の埋め込みとの比較）:")
    print(quality_table(report))
    print()
    print(f"サイズ試算（対象テンソル集合のみ・N = 要素数・G = N/{GROUP_SIZE}）:")
    print(size_table(report))
    print()
    print(f"report → {args.out / REPORT_FILE}")
    failures = report["gates"]["failures"]
    if failures:
        raise AssertionError("検証ゲートが赤: " + " / ".join(failures))
    print("gates: all green")


if __name__ == "__main__":
    main()
