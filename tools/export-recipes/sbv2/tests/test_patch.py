"""SBV2 パッチ層の単体テスト（M1-P3 波 6）。

各パッチは「原実装と eager 同値」であることが存在条件（ADR 0013）。実重みを使う貫通検証は
`test_export.py` と `sbv2.export --verify` が受け持ち、ここは**合成の小さいモジュール**
で個別のパッチを 1 つずつ切り分ける。`style_bert_vits2` パッケージが無い環境では SKIP。

MUST: クラス属性を差し替えるテストは `pristine_classes` フィクスチャを取る。差し替えは
プロセス全域に効くので、①入口で**原実装へ戻さない**と「原実装のつもりでパッチ後を掴む」
恒真な比較になり、②出口で元へ戻さないとこの後のテストが黙って別の実装を検証する。
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from sbv2 import patch as patch_sbv2

_PACKAGE_PRESENT = importlib.util.find_spec("style_bert_vits2") is not None

requires_package = pytest.mark.skipif(
    not _PACKAGE_PRESENT,
    reason="style_bert_vits2 が無い（`uv sync --group sbv2` が前提）",
)

#: 同値判定の許容差。パッチは「同じ式を別の順序で組む」書き換えなので、差は f32 の
#: 縮約順序差（数 ulp）に留まるはず。値域 O(1) に対して 1e-5 なら実装の取り違え
#: （添字ずれ・軸取り違え）の O(1) 誤差とは 5 桁離れている。
EQUIVALENCE_ATOL = 1e-5

#: **収集時に採った原実装**。pytest は全テストモジュールを収集してから実行するので、
#: ここはどのテストも走っていない = まだ誰もパッチを当てていない時点になる。
#:
#: MUST: 「テスト開始時の値」を退避するだけでは足りない — 同一プロセスの別テスト
#: （`test_export.py` の front export）が先にパッチを当てていると、原実装のつもりで
#: **パッチ後の実装**を掴み、同値テストが patched vs patched の**恒真な比較**になる
#: （実際にファイル単独では緑・全体実行でも緑、という形で一度これを踏んだ）。
if _PACKAGE_PRESENT:
    from style_bert_vits2.models import attentions as _sbv2_attentions
    from style_bert_vits2.models import modules as _sbv2_modules

    PRISTINE_MHA_FORWARD = _sbv2_attentions.MultiHeadAttention.forward
    PRISTINE_FFN_FORWARD = _sbv2_attentions.FFN.forward
    PRISTINE_SPLINE = _sbv2_modules.piecewise_rational_quadratic_transform


@pytest.fixture
def pristine_classes():
    """パッチ対象を**原実装**に戻してからテストへ入り、抜けるとき元の状態へ復す。

    入口で原実装へ戻すのは恒真化の防止（上の MUST）。出口で「入ってきたときの状態」へ
    復すのは、このテストが他のテストの前提（front の export はパッチ済みを前提にする）を
    壊さないため — 入口と出口で戻す先が違うのは意図的。
    """
    from style_bert_vits2.models import modules
    from style_bert_vits2.models.attentions import FFN, MultiHeadAttention

    entered = (
        MultiHeadAttention.forward,
        FFN.forward,
        modules.piecewise_rational_quadratic_transform,
        patch_sbv2.patches_applied(),
    )
    MultiHeadAttention.forward = PRISTINE_MHA_FORWARD
    FFN.forward = PRISTINE_FFN_FORWARD
    modules.piecewise_rational_quadratic_transform = PRISTINE_SPLINE
    patch_sbv2._APPLIED = False
    try:
        yield
    finally:
        MultiHeadAttention.forward = entered[0]
        FFN.forward = entered[1]
        modules.piecewise_rational_quadratic_transform = entered[2]
        patch_sbv2._APPLIED = entered[3]


def _spline_arguments(seed: int, num_bins: int = 10, tail_bound: float = 5.0):
    """ConvFlow が spline に渡す形 `[b, c, t, ?]` の合成入力。

    `inputs` は tail_bound の**外側も内側も**踏む値域にする（区間外を入力そのままへ戻す
    経路と spline 本体の両方を 1 ケースで通すため）。
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 1, 7)
    inputs = torch.randn(shape, generator=generator) * tail_bound
    return {
        "inputs": inputs,
        "unnormalized_widths": torch.randn((*shape, num_bins), generator=generator),
        "unnormalized_heights": torch.randn((*shape, num_bins), generator=generator),
        "unnormalized_derivatives": torch.randn((*shape, num_bins - 1), generator=generator),
        "tail_bound": tail_bound,
    }


@requires_package
class TestSplineEquivalence:
    """① spline の分岐フリー・非破壊化。"""

    @pytest.mark.parametrize("inverse", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_it_matches_the_original_transform(self, inverse, seed):
        from style_bert_vits2.models import transforms

        args = _spline_arguments(seed)
        tail_bound = args.pop("tail_bound")
        # 原実装は unnormalized_derivatives を in-place で書き換えるので clone を渡す
        # （渡さないと 2 回目以降の比較が別の入力になる）。
        expected = transforms.piecewise_rational_quadratic_transform(
            **{key: value.clone() for key, value in args.items()},
            inverse=inverse,
            tails="linear",
            tail_bound=tail_bound,
        )
        got = patch_sbv2.unconstrained_rqs_free(
            **{key: value.clone() for key, value in args.items()},
            inverse=inverse,
            tails="linear",
            tail_bound=tail_bound,
        )

        for index, (a, b) in enumerate(zip(got, expected, strict=True)):
            assert torch.allclose(a, b, atol=EQUIVALENCE_ATOL, rtol=0), index

    def test_the_case_actually_crosses_the_tail_bound(self):
        """恒真化の門番 — 区間内と区間外の**両方**を含む入力でないと where 経路が空振りする。"""
        args = _spline_arguments(0)
        inside = (args["inputs"].abs() <= args["tail_bound"]).sum().item()

        assert 0 < inside < args["inputs"].numel()

    def test_it_does_not_destroy_its_arguments(self):
        """非破壊化そのものの検査（原実装は `bin_locations[..., -1] += eps` で破壊する）。"""
        args = _spline_arguments(0)
        tail_bound = args.pop("tail_bound")
        before = {key: value.clone() for key, value in args.items()}

        patch_sbv2.unconstrained_rqs_free(
            **args, inverse=True, tails="linear", tail_bound=tail_bound
        )

        for key, value in args.items():
            assert torch.equal(value, before[key]), key

    def test_searchsorted_matches_the_original(self):
        from style_bert_vits2.models import transforms

        generator = torch.Generator().manual_seed(3)
        # 実使用と同じ形: 境界は [b, c, t, bins+1]、値は [b, c, t]（最終軸だけが bin 軸）。
        locations = torch.cumsum(torch.rand(2, 5, 3, 6, generator=generator), dim=-1)
        values = torch.rand(2, 5, 3, generator=generator) * float(locations.max())

        got = patch_sbv2.searchsorted_free(locations, values)
        # 原実装は第 1 引数を破壊するので、比較用に clone を渡す（破壊された列で比べると
        # 「同じ破壊を受けた者どうし」の恒真な比較になる）。
        expected = transforms.searchsorted(locations.clone(), values)

        assert torch.equal(got, expected)

    def test_tails_none_is_rejected_after_patching(self, pristine_classes):
        """`tails=None` は原実装の定義域検査を外した spline へ直行する経路なので受けない。"""
        from style_bert_vits2.models import modules

        patch_sbv2.apply_spline_patch()
        args = _spline_arguments(0)
        args.pop("tail_bound")

        with pytest.raises(RuntimeError, match="tails=None"):
            modules.piecewise_rational_quadratic_transform(**args, tails=None)

    def test_the_patch_replaces_the_symbol_modules_actually_calls(self, pristine_classes):
        """差し替え先は modules（transforms 側を替えても modules の束縛には効かない）。"""
        from style_bert_vits2.models import modules, transforms

        original = modules.piecewise_rational_quadratic_transform
        assert original is transforms.piecewise_rational_quadratic_transform

        patch_sbv2.apply_spline_patch()

        assert modules.piecewise_rational_quadratic_transform is not original
        assert transforms.piecewise_rational_quadratic_transform is original


def _attention(seed: int, window_size: int = 4, channels: int = 24, heads: int = 2):
    from style_bert_vits2.models.attentions import MultiHeadAttention

    torch.manual_seed(seed)
    attn = MultiHeadAttention(channels, channels, heads, window_size=window_size)
    attn.eval()
    return attn


def _attention_case(length: int, channels: int = 24, seed: int = 11):
    generator = torch.Generator().manual_seed(seed + length)
    x = torch.randn(1, channels, length, generator=generator)
    mask = torch.ones(1, 1, length)
    attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
    return x, attn_mask


@requires_package
class TestGatherRelativeAttention:
    """② 相対位置注意の gather 化。"""

    @pytest.mark.parametrize("length", [2, 4, 5, 9, 17])
    def test_it_matches_the_original_forward(self, pristine_classes, length):
        """窓幅 4 に対し P < w+1 / P == w+1 / P > w+1 の 3 領域を踏む。

        原実装の `_get_relative_embeddings` は `P ≤ window_size` で埋め込みを**別の位置から
        切り出す**（pad するか slice するかが切り替わる）ため、大きい P だけで比べると
        小さい P の切り出し規則の取り違えを見逃す。
        """
        from style_bert_vits2.models.attentions import MultiHeadAttention

        # 恒真化の門番: 参照値は**原実装**で採る（フィクスチャが保証するが、外れたら
        # patched vs patched の比較になって静かに意味が消えるのでここでも見る）。
        assert MultiHeadAttention.forward is PRISTINE_MHA_FORWARD
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(length)
        with torch.no_grad():
            expected = attn(x, x, attn_mask)

        patch_sbv2.apply_gather_relattn_patch()
        with torch.no_grad():
            got = attn(x, x, attn_mask)

        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_the_relative_embeddings_actually_move_the_result(self, pristine_classes):
        """恒真化の門番 — 相対位置の埋め込みを変えれば出力も変わる。

        これが無いと、上の同値テストは「gather 経路の寄与が丸ごと 0」でも緑になる
        （実重みの emb_rel_* は小さいので、寄与が消えていても値がそれらしく見える）。
        """
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(9)
        patch_sbv2.apply_gather_relattn_patch()
        with torch.no_grad():
            before = attn(x, x, attn_mask)
            attn.emb_rel_k.add_(1.0)
            attn.emb_rel_v.add_(1.0)
            after = attn(x, x, attn_mask)

        assert not torch.allclose(before, after, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_the_masked_columns_are_excluded(self, pristine_classes):
        """attn_mask=0 の列が softmax から外れる（マスク経路が gather 化で消えていない）。"""
        attn = _attention(seed=5)
        x, _ = _attention_case(9)
        mask = torch.ones(1, 1, 9)
        mask[..., 6:] = 0.0
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        with torch.no_grad():
            expected = attn(x, x, attn_mask)

        patch_sbv2.apply_gather_relattn_patch()
        with torch.no_grad():
            got = attn(x, x, attn_mask)

        assert torch.allclose(got, expected, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_heads_share_false_is_rejected(self, pristine_classes):
        """head ごとに別の埋め込みを持つ構成は、head 軸 1 本の expand では表せない。"""
        attn = _attention(seed=5)
        attn.heads_share = False
        x, attn_mask = _attention_case(9)
        patch_sbv2.apply_gather_relattn_patch()

        with pytest.raises(ValueError, match="heads_share"):
            attn(x, x, attn_mask)

    def test_a_window_free_attention_is_rejected(self, pristine_classes):
        attn = _attention(seed=5)
        attn.window_size = None
        x, attn_mask = _attention_case(9)
        patch_sbv2.apply_gather_relattn_patch()

        with pytest.raises(ValueError, match="window_size"):
            attn(x, x, attn_mask)


@requires_package
class TestRelattnTableThreading:
    """② の表スレッディング（flow / voice 経路 — 表をグラフ入力から注意層まで運ぶ形）。"""

    @pytest.mark.parametrize("length", [2, 4, 5, 9, 17])
    def test_supplied_tables_match_the_original_forward(self, pristine_classes, length):
        """外部供給の表で組んだ注意が**原実装**と同値（front と同じ 3 領域を踏む）。"""
        from style_bert_vits2.models.attentions import MultiHeadAttention

        assert MultiHeadAttention.forward is PRISTINE_MHA_FORWARD  # 恒真化の門番
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(length)
        with torch.no_grad():
            expected = attn(x, x, attn_mask)

        tables = patch_sbv2.build_relattn_tables(length)
        with torch.no_grad():
            got = patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, tables)

        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=EQUIVALENCE_ATOL, rtol=0)

    @pytest.mark.parametrize("length", [2, 5, 17])
    def test_the_two_table_sources_are_bit_identical(self, pristine_classes, length):
        """front（in-graph 構築 = 焼き込み）と flow（グラフ入力）が**同じ数**を出す。

        表の式は 1 箇所（`build_relattn_tables`）にしか無い、という設計の実測。ここが
        割れると、front は焼き込んだ表で・flow はホストが作った表で、**どちらも
        shape は合ったまま別のモデル**になる（ADR 0013 の沈黙誤値クラス）。
        """
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(length)
        with torch.no_grad():
            in_graph = patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, None)
            supplied = patch_sbv2.mha_gather_forward(
                attn, x, x, attn_mask, patch_sbv2.build_relattn_tables(length)
            )

        assert torch.equal(in_graph, supplied)

    def test_a_table_shifted_by_one_changes_the_result(self, pristine_classes):
        """故障注入 — 添字表を 1 ずらすと出力が変わる。

        窓幅の取り違えは `idx_k` を 1 ずらすのと同じ形で、**要素数は合うので shape
        エラーにならない**。上の同値テストがこのクラスの誤りを実際に検出できること
        （= 表の寄与が丸ごと 0 に潰れていないこと）をここで固定する。
        """
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(17)
        idx_k, valid = patch_sbv2.build_relattn_tables(17)
        window = 2 * 4
        shifted = torch.clamp(idx_k + 1, 0, window)
        with torch.no_grad():
            correct = patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, (idx_k, valid))
            wrong = patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, (shifted, valid))

        assert not torch.allclose(correct, wrong, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_a_wider_window_runs_off_the_embedding(self, pristine_classes):
        """窓幅が**大きい**側へずれた表は添字が埋め込みの外へ出て torch が落ちる。

        小さい側へずれた場合は範囲内の別の行を読むので黙って誤る（上の 1 ずらしが
        その形）。両側で挙動が違うことを記録しておかないと「落ちなかった＝正しい」と
        読み違える。
        """
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(17)
        tables = patch_sbv2.build_relattn_tables(17, window_size=5)

        with pytest.raises(RuntimeError):
            patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, tables)


class TestRelattnTableBuilder:
    """表生成器そのもの
    （ホスト TS 鏡像 `packages/models/src/sbv2/relattn-tables.ts` の正本）。
    """

    def test_the_tables_have_the_square_shape_and_dtypes(self):
        idx_k, valid = patch_sbv2.build_relattn_tables(7)

        assert idx_k.shape == (7, 7)
        assert valid.shape == (7, 7)
        assert idx_k.dtype == torch.int64
        assert valid.dtype == torch.float32

    def test_the_indices_stay_inside_the_embedding(self):
        """clamp 済みなので gather の範囲外規約（NaN 汚染）には抵触しない（ADR 0013）。"""
        idx_k, _ = patch_sbv2.build_relattn_tables(64)

        assert int(idx_k.min()) == 0
        assert int(idx_k.max()) == 2 * 4

    def test_the_window_mask_matches_the_relative_distance(self):
        length, window = 12, 4
        _, valid = patch_sbv2.build_relattn_tables(length, window)
        positions = torch.arange(length)
        expected = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() <= window

        assert torch.equal(valid, expected.to(torch.float32))

    def test_a_different_window_gives_different_tables(self):
        """恒真化の門番（TS 側パリティテストと対）— 窓幅が効いていることの実測。"""
        narrow = patch_sbv2.build_relattn_tables(9, 4)
        wide = patch_sbv2.build_relattn_tables(9, 5)

        assert narrow[0].shape == wide[0].shape
        assert not torch.equal(narrow[0], wide[0])
        assert not torch.equal(narrow[1], wide[1])


@requires_package
class TestFfnPaddingFold:
    """③ FFN の明示 pad → conv の padding 引数。"""

    @pytest.mark.parametrize("kernel_size", [1, 3, 5])
    def test_it_matches_the_original_forward_for_odd_kernels(self, pristine_classes, kernel_size):
        from style_bert_vits2.models.attentions import FFN

        assert FFN.forward is PRISTINE_FFN_FORWARD  # 恒真化の門番（上の gather 版と同じ）
        torch.manual_seed(7)
        ffn = FFN(16, 16, 32, kernel_size)
        ffn.eval()
        generator = torch.Generator().manual_seed(8)
        x = torch.randn(1, 16, 11, generator=generator)
        mask = torch.ones(1, 1, 11)
        with torch.no_grad():
            expected = ffn(x, mask)

        patch_sbv2.apply_ffn_conv_padding_patch()
        with torch.no_grad():
            got = ffn(x, mask)

        assert torch.allclose(got, expected, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_an_even_kernel_is_rejected(self, pristine_classes):
        """偶数 kernel の `_same_padding` は左右非対称で、conv の padding と等価でない。"""
        from style_bert_vits2.models.attentions import FFN

        torch.manual_seed(7)
        ffn = FFN(16, 16, 32, 4)
        ffn.eval()
        patch_sbv2.apply_ffn_conv_padding_patch()

        with pytest.raises(ValueError, match="偶数"):
            ffn(torch.randn(1, 16, 11), torch.ones(1, 1, 11))

    def test_a_causal_ffn_is_rejected(self, pristine_classes):
        from style_bert_vits2.models.attentions import FFN

        torch.manual_seed(7)
        ffn = FFN(16, 16, 32, 3, causal=True)
        ffn.eval()
        patch_sbv2.apply_ffn_conv_padding_patch()

        with pytest.raises(ValueError, match="causal"):
            ffn(torch.randn(1, 16, 11), torch.ones(1, 1, 11))

    @pytest.mark.parametrize("kernel_size", [2, 4])
    def test_an_even_kernel_really_differs(self, kernel_size):
        """恒真化の門番 — 偶数 kernel では原実装のパディングと conv の padding が実際に違う。

        原実装 `_same_padding` は左 `(k−1)//2` / 右 `k//2` の**非対称** pad で、conv1d の
        padding 引数（左右同数）では作れない。上の拒否 assert が「念のため」ではなく実害を
        止めていることの実測（偶数では出力長からしてずれる）。
        """
        from torch.nn import functional

        x = torch.zeros(1, 1, 11)
        explicit = functional.pad(x, [(kernel_size - 1) // 2, kernel_size // 2])
        symmetric = functional.pad(x, [(kernel_size - 1) // 2] * 2)

        assert explicit.shape != symmetric.shape


class TestConstantColumn:
    """定数列の作り方（`full_like` を IR に持ち込まないための定型手筋）。"""

    def test_it_has_the_leading_shape_with_a_single_trailing_column(self):
        source = torch.randn(2, 3, 4)

        column = patch_sbv2._constant_column(source, 1.5)

        assert column.shape == (2, 3, 1)
        assert torch.equal(column, torch.full((2, 3, 1), 1.5))

    def test_a_negative_zero_source_still_yields_the_exact_constant(self):
        """`x * 0.0` は x = −0.0 でも −0.0 で、`+ value` が厳密に value を返す。"""
        source = torch.full((1, 1, 2), -0.0)

        assert torch.equal(patch_sbv2._constant_column(source, -5.0), torch.full((1, 1, 1), -5.0))


@requires_package
class TestPatchApplicationFlag:
    """パッチのプロセス汚染を可視化するフラグ（恒真化の遮断に使う）。"""

    def test_it_flips_only_after_applying(self, pristine_classes):
        # フィクスチャが未適用状態にして入る。
        assert not patch_sbv2.patches_applied()

        patch_sbv2.apply_all_patches()

        assert patch_sbv2.patches_applied()


@requires_package
class TestWrapperShapes:
    """export 用ラッパの形（実重みを使わない範囲の固定）。"""

    def test_the_forward_signature_is_the_ir_input_order(self):
        import inspect

        from sbv2 import export as export_sbv2

        names = list(inspect.signature(patch_sbv2.Sbv2Front.forward).parameters)[1:]

        # IR の入力名は forward の引数名がそのまま出る。台本側の並びと食い違うと
        # dynamic_shapes が別の入力に付く（P が付かない入力は静的に焼かれて壊れる）。
        assert names == list(export_sbv2.FRONT_INPUT_ORDER)

    def test_the_sdp_wrapper_takes_the_noise_as_an_input(self):
        import inspect

        names = list(inspect.signature(patch_sbv2.SdpReverseNoiseIn.forward).parameters)[1:]

        assert names == ["x", "x_mask", "g", "z_noise"]

    @pytest.mark.parametrize("wrapper", [patch_sbv2.FlowReverse, patch_sbv2.Sbv2Voice])
    def test_the_flow_family_takes_the_tables_as_inputs(self, wrapper):
        """flow / voice は表を**グラフ入力**で受ける（front の焼き込みと違う — ADR 0013）。"""
        import inspect

        from sbv2 import export as export_sbv2

        names = list(inspect.signature(wrapper.forward).parameters)[1:]

        assert names == list(export_sbv2.FLOW_INPUT_ORDER)
        assert names[-2:] == ["idx_k", "valid"]


# NOTE: 添字表そのもの（`idx_k = clamp(w + j − i, 0, 2w)` / `idx_v = i + c`）を式のまま
# 再掲して検査するテストは**置かない** — パッチのコードを写した式を検査しても、写しの側が
# 正しいことしか言えない（原実装との突合こそが検査）。向きの取り違えも範囲外添字も、上の
# `test_it_matches_the_original_forward` が拾う（範囲外の gather は torch 自身が例外にする）。
