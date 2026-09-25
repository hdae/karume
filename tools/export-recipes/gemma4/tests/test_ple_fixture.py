"""リポジトリに置いた packed PLE fixture が、今の生成器を回した結果と**バイト同一**であること。

読み手側（`packages/models/tests/gemma4_ple_packed_test.ts`）は fixture の**中身**を torch 参照
とビット一致で見ているが、「生成器 → 現物」の対応は手で回すことしか繋いでいなかった。本番経路
（{@link karume.ple.ple_assets} / {@link karume.pipeline.publish_model}）が block の切り方や
容器のバイト配置を変えても、誰も赤くならないまま fixture だけが古い形で残る。

比べるのは `krm`（容器）だけである。隣の `oracle.safetensors` は metadata に `torch.__version__`
を焼くので、torch を上げた機ではバイトが動く — そこまで縛ると「torch を上げたら赤」になり、
門が守りたいもの（容器の形）と無関係な理由で落ちる。
"""

from __future__ import annotations

from pathlib import Path

from gemma4.tests.ple_fixture import FIXTURE_ROOT, write_fixture

#: 比べる格納 dtype（生成器が焼く全部）。
STORAGES = ("i2", "i4")


def _parts(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.glob("*.krm"))}


class TestTheCommittedFixtureMatchesItsGenerator:
    """`uv run python -m gemma4.tests.ple_fixture` を回した結果が現物と一致すること。"""

    def test_every_container_part_is_byte_identical(self, tmp_path: Path) -> None:
        write_fixture(tmp_path)

        for storage in STORAGES:
            committed = _parts(FIXTURE_ROOT / storage)
            regenerated = _parts(tmp_path / storage)

            # 非恒真: 現物が空（= 比較が 0 件）なら落とす。分割形なので part は 2 本以上。
            assert len(committed) >= 2, f"{storage}: リポジトリ側の krm が {len(committed)} 本"
            assert sorted(regenerated) == sorted(committed), storage
            for name, payload in committed.items():
                assert regenerated[name] == payload, (
                    f"{storage}/{name} が生成器の出力と違う —"
                    " `cd tools/export-recipes && uv run python -m gemma4.tests.ple_fixture`"
                    " で焼き直す"
                )
