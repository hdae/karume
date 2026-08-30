"""視認評価専用の組み立て（`anima/eval_dist.py`）の約束事。

ここが守るのは 1 点だけ — **配布と評価を取り違えない**こと。i4 の感度実験変種は格納形も
本数も `verify_dist` もヘッダ検査も既定と同じ（i4 と i8 の混成）なので、資産から見分ける手は
`calib_provenance.json` の `method` しか無い。したがって:

- 評価の口は**変種の記録しか受けない**（配布条件で焼いた系列を挿しても落ちる）
- 配布の口は**既定の記録しか受けない**（変種を挿しても落ちる）

の**両方向**が要る。片方だけだと「i4 席の中身がどちらの丸め方か」が置き場から読めなくなる。

組み立てエンジンそのもの（配置・共有・manifest・検証）は `test_distribution.py` の担当なので、
ここでは計画（1 バイトも書かない側）までを見る。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ir_fixtures import ir_container

from anima import eval_dist
from anima.distribution import (
    ADALN_I8_CALIB_METHOD,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_TURBO_MODEL_NAME,
    CALIB_PROVENANCE_FILE,
    CALIB_SHIPPABLE_METHOD,
    anima_model,
    anima_sources,
    assert_calib_provenance,
)
from karume.dist import DistError

#: 素版の視認対象（i4 席が配布から外れているモデル — 2026-08-24 の裁定）。
MODEL = ANIMA_BASE_MODEL_NAME


def _fake_safetensors(dtype: str, payload: bytes) -> bytes:
    """格納 dtype の門を通る最小の safetensors（8 バイト長 + ヘッダ JSON + データ節）。"""
    header: dict[str, Any] = {
        "w": {"dtype": dtype, "shape": [len(payload)], "data_offsets": [0, len(payload)]}
    }
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + payload


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _calib_record(method: str, model: str) -> bytes:
    """`anima/export.py` が i4 系列へ残す校正条件の記録（実物と同じ形）。

    step / CFG は**モデル別**（`anima.calib.calib_conditions` が `pipeline_config` から導く）
    ので、身代わりも同じ 1 箇所から引く — 写すと「どのモデルの条件で焼いたか」を見る門と
    独立でなくなる。
    """
    defaults = anima_model(model).pipeline_config["defaults"]
    return json.dumps(
        {
            "method": method,
            "group_size": 32,
            "grid": "rtn",
            "prompts": 4,
            "resolution": 512,
            "steps": int(defaults["steps"]),
            "guidance": float(defaults["guidanceScale"]),
            "text_dtype": "f16",
        }
    ).encode("utf-8")


def _build_series(series_dir: Path, *, method: str = ADALN_I8_CALIB_METHOD) -> Path:
    """素版 1 モデルの系列（f16 / i8 の配布系列 + **変種の i4 系列**）を偽資産で作る。

    weights の席だけは**正当な IR コンテナ**（組み立ては入力を IR v1 の全規則で見る —
    `karume.dist.assert_weight_components_verified`）。rope 素表は extras の席なので
    IR コンテナではなく、従来どおりヘッダだけの偽資産でよい。
    """
    sources = anima_sources(series_dir, MODEL)
    _write(
        sources.base / "text_encoder" / "model.safetensors",
        ir_container(mark="te", storage="f16"),
    )
    _write(
        sources.text_conditioner / "text_conditioner" / "model.safetensors",
        ir_container(mark="tc", storage="f16"),
    )
    _write(
        sources.base / "vae_decoder" / "model.safetensors",
        ir_container(mark="vae", storage="f16"),
    )
    _write(sources.tokenizers / "qwen2-tokenizer.json", b'{"qwen2": true}')
    _write(sources.tokenizers / "t5-tokenizer.json", b'{"t5": true}')
    rope = _fake_safetensors("F32", b"rope")
    variant = series_dir / eval_dist.EVAL_SERIES.format(model=MODEL)
    for series, storage in (
        (sources.transformer["f16"], "f16"),
        (sources.transformer["i8"], "i8"),
        (variant, "i4"),
    ):
        _write(
            series / "transformer" / "model.safetensors",
            ir_container(mark=f"dit-{storage}", storage=storage),
        )
        _write(series / "transformer" / "rope_base.safetensors", rope)
    _write(variant / "transformer" / CALIB_PROVENANCE_FILE, _calib_record(method, MODEL))
    return variant


class TestEvalPlan:
    def test_it_restores_the_i4_seat_without_touching_the_shipping_table(self, tmp_path):
        """MUST: 席は spec の差し替えで戻す — `ANIMA_MODELS` は 1 行も動かない。"""
        _build_series(tmp_path / "series")

        plan = eval_dist.eval_plan(tmp_path / "series", MODEL)

        assert "i4" not in anima_model(MODEL).storages, "配布の受理集合が動いている"
        assert set(plan.weights["transformer"]) == {"f16", "i8", "i4"}
        assert "f16+dit4" in plan.quants

    def test_the_i4_seat_reads_the_variant_series(self, tmp_path):
        """MUST: 配布条件で焼いた `*-i4-dyn` は見に行かない（同じ席へ別の丸め方が入らない）。"""
        variant = _build_series(tmp_path / "series")

        plan = eval_dist.eval_plan(tmp_path / "series", MODEL)

        assert (
            plan.artifacts["transformer_i4"].source == variant / "transformer" / "model.safetensors"
        )
        assert eval_dist.EVAL_SERIES.format(model=MODEL) != f"{MODEL}-i4-dyn"

    def test_a_series_calibrated_for_shipping_is_refused(self, tmp_path):
        """取り違えの片方向: 既定の丸め方で焼いた系列を視認の席へ挿しても落ちる。"""
        _build_series(tmp_path / "series", method=CALIB_SHIPPABLE_METHOD)

        with pytest.raises(DistError, match="配布して良い丸め方式で作られていない"):
            eval_dist.eval_plan(tmp_path / "series", MODEL)

    def test_a_series_without_calibration_is_refused(self, tmp_path):
        """`--no-calib` の smoke 生成物で視認裁定を採らない（品質の基線が別物）。"""
        _build_series(tmp_path / "series", method="rtn-adaln8")

        with pytest.raises(DistError):
            eval_dist.eval_plan(tmp_path / "series", MODEL)

    def test_turbo_is_refused(self, tmp_path):
        """turbo は配布で i4 席を持つ — 視認したいなら `dist.py` の配布経路がそのまま使える。"""
        with pytest.raises(DistError, match="視認評価の対象外"):
            eval_dist.eval_plan(tmp_path / "series", ANIMA_TURBO_MODEL_NAME)

    def test_the_default_output_never_lands_in_the_distribution_root(self):
        """MUST: `models/` は配布形だけの場所（`_shared.paths` の DECIDED）。"""
        out = eval_dist.default_out_dir(eval_dist.PIPELINE, [MODEL])

        assert out.parent == eval_dist.EVAL_ROOT
        assert "models" not in out.parts

    def test_more_than_one_model_needs_an_explicit_output(self):
        with pytest.raises(DistError, match="--out"):
            eval_dist.default_out_dir(eval_dist.PIPELINE, [MODEL, "anima-wai-v1.0"])


class TestEndToEnd:
    """CLI から 1 周（計画 → 実体化 → カード → `verify_dist`）通ることを見る。

    ここだけ実体化まで踏むのは費用の非対称のため — 変種の i4 系列を焼くのに実測 ~3h 掛かる
    ので、組み立て側の綻び（カードの描き手・既定の出力先・manifest の形）を**焼いた後で**
    知る形にしない。
    """

    def test_the_cli_assembles_a_loadable_distribution(self, tmp_path):
        _build_series(tmp_path / "series")
        out_dir = tmp_path / "eval" / "karume-anima-v1.0-adaln8"

        eval_dist.main(["--series", str(tmp_path / "series"), "--out", str(out_dir)])

        manifest = json.loads((out_dir / "karume.json").read_text(encoding="utf-8"))
        assert manifest["defaultModel"] == MODEL
        quants = manifest["models"][MODEL]["quants"]
        assert "f16+dit4" in quants
        assert quants["f16+dit4"]["weights"]["transformer"] == "i4"
        # 視認の入口（`examples/anima/main.ts` の `--source`）が要求するのは manifest 1 枚。
        assert (out_dir / "README.md").is_file()


class TestShippingRefusesTheVariant:
    """取り違えのもう片方向: 変種の記録は**配布の**組み立てで落ちる。

    素版は i4 席そのものが配布から外れているので、この網が効くのは i4 席を持つ turbo。
    席が将来素版へ戻っても同じ 1 実装が掛かる（判定は `spec.calib_method` 1 箇所）。
    """

    def test_a_variant_record_fails_the_shipping_gate(self, tmp_path):
        model = ANIMA_TURBO_MODEL_NAME
        sources = anima_sources(tmp_path / "series", model)
        _write(
            sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE,
            _calib_record(ADALN_I8_CALIB_METHOD, model),
        )

        with pytest.raises(DistError, match="配布して良い丸め方式で作られていない"):
            assert_calib_provenance(sources, anima_model(model))

    def test_the_shipping_gate_still_accepts_the_default_record(self, tmp_path):
        model = ANIMA_TURBO_MODEL_NAME
        sources = anima_sources(tmp_path / "series", model)
        _write(
            sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE,
            _calib_record(CALIB_SHIPPABLE_METHOD, model),
        )

        assert_calib_provenance(sources, anima_model(model))
