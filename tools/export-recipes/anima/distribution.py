"""Anima の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **Anima 固有の事実**だけ: どの系列ディレクトリから
何を拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

公開面は Pipeline 2 つ（{@link OFFICIAL_PIPELINE} / {@link EXTRA_PIPELINE} — ADR 0087）—
リポの dist ドライバ（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from _shared.calib_provenance import calib_complaint
from anima.card import (
    ATTRIBUTION_NOTICE,
    render_base_card,
    render_extra_card,
)
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
    sha256_file,
)

#: 公式 Turbo 変種のモデル名（= 公式リポの既定モデル — 上流 README が「まず Turbo を」と
#: 推奨・2026-09-01 裁定）。上流の名乗りをそのまま使う（ADR 0077）。旧 `anima-turbo`
#: （turbo LoRA v0.2 を焼き込んだ配布物・hdae/karume-anima-turbo）はこの公式 checkpoint 版で
#: 置き換えられ、受理集合から退役した（2026-09-01 — 焼き込み由来の旧系列
#: `outputs/series/anima-turbo-*-dyn` は `lora_provenance.json` を持つので、誤って新しい席へ
#: 挿すと {@link assert_lora_provenance} が落とす）。
ANIMA_TURBO_MODEL_NAME = "anima-turbo-v1.1"

#: 素の base 配布物のモデル名。多 step + CFG で使う席で、negative prompt が効くのは
#: この系（base / aesthetic / extra）だけ（turbo は CFG=1 運用なので uncond 側を計算しない）。
ANIMA_BASE_MODEL_NAME = "anima-v1.0"

#: 公式 Aesthetic 変種のモデル名（base を高品質画風へ fine-tune した公式版 — 上流の名乗り
#: そのまま）。text_conditioner は base と f16 丸め後も 64 テンソル差が残る実測
#: （2026-09-01）から自前で持つ。
ANIMA_AESTHETIC_MODEL_NAME = "anima-aesthetic-v1.1"

#: パイプライン契約（ADR 0041 §2 — モデル単位）。
ANIMA_PIPELINE = "anima/1"

#: モデル名に依らない系列（base の text 経路 / VAE と、tokenizer を書く台本の出力）。
#: MUST: `anima-f16` は **LoRA を焼かずに** 焼いた系列である（`lora_provenance.json` が
#: 無いことがその記録）。turbo の text_conditioner 成分は実測で lora_B が全ゼロ = noop
#: （`anima/pipeline_ref.py` の同 NOTE）なので、turbo 席もここを共有して差し支えない。
ANIMA_BASE_SERIES = "anima-f16"
ANIMA_TOKENIZER_SERIES = "anima-demo"

#: 焼き込んだ LoRA の帰属を残すファイル（系列のターゲット直下）。系列レイアウトの綴りは
#: 読み手（ここ）が持ち、書き手（`anima/export.py`）はここから引く — 2 箇所で独立に動かさない。
LORA_PROVENANCE_FILE = "lora_provenance.json"

#: i4 系列が記録する校正条件（{@link assert_calib_provenance}）。校正の有無は**格納形を 1 バイトも
#: 変えない**ので、資産・manifest・ヘッダのどれからも判別できない。それでいて品質差は視認裁定を
#: 分ける大きさ（素の RTN =「全体的にぼやけた印象」/ GPTQ =「高品質を維持」— research 2026-08-21
#: §6）なので、LoRA 帰属と同じ規律で「書き出した側が事実を書き残す」。
CALIB_PROVENANCE_FILE = "calib_provenance.json"

#: 配布して良い丸め方式（{@link CALIB_PROVENANCE_FILE} の `method`）。`--no-calib` の素の RTN は
#: smoke 用で、配布資産にしない（`anima/export.py` の該当 MUST）。
CALIB_SHIPPABLE_METHOD = "gptq"

#: i4 の量子化感度実験変種（block 内 adaLN + block 外を i8 格納へ）の綴り。系列ディレクトリの
#: 接尾辞と {@link CALIB_PROVENANCE_FILE} の `method` 接尾辞を**同じ 1 語**から作る。
#: 書き手は `anima/export.py` の `--i4-adaln-i8`（綴りは LoRA / 校正の記録と同じ規律で読み手が
#: 持つ — 2 箇所で独立に動かさない）。
ADALN_I8_TAG = "adaln8"

#: 同・変種が記録する丸め方式。**{@link CALIB_SHIPPABLE_METHOD} と一致しない**綴りであることが
#: 要（格納形も本数も `verify_dist` もヘッダ検査も素通りする変種なので、配布経路から締め出す
#: 網は {@link assert_calib_provenance} の方式一致 1 つきり）。これを要求するのは視認評価専用の
#: 組み立て（`anima.eval_dist`）だけ。
ADALN_I8_CALIB_METHOD = f"{CALIB_SHIPPABLE_METHOD}-{ADALN_I8_TAG}"

#: GPTQ 校正（捕捉 + 丸め）を回せるデバイス（書き手は `anima/export.py` の `--calib-device`）。
CALIB_DEVICES = ("cpu", "cuda")

#: **配布して良い校正デバイス**。これ以外で焼いた i4 は {@link CALIB_PROVENANCE_FILE} の `method`
#: へデバイス名が付き（`gptq-cuda`）、{@link assert_calib_provenance} の方式一致に名指しで落ちる。
#: 綴りを変えるのは {@link ADALN_I8_TAG} と同じ理由 — **GPTQ の丸め解はデバイスで変わる**
#: （f64 縮約の順序も linalg の実装も違う）のに、格納形も本数も 1 バイト違わないので
#: `verify_dist` もヘッダ検査も素通りする。配布経路から締め出す網は方式一致 1 つきり。
CALIB_SHIPPABLE_DEVICE = "cpu"

#: 上流の重みライセンス原文（この recipe の隣に逐語で置いてある）。配布は Derivative の
#: Distribution なので §3(a)（ライセンスのコピーを第三者へ提供する）が掛かる — 要約や
#: 書き換えでは条件を満たさないため、**1 バイトも変えずに**配布リポ直下の `LICENSE.md` として
#: 出す。`Path(__file__)` 基準で引くのは、cwd にも系列の置き場にも依存しないため。
LICENSE_SOURCE_PATH = Path(__file__).parent / "circlestone_license.txt"

#: どのリポの告知にも載る改変（コンテナ形式への変換）。
#:
#: MUST: 改変の列挙は**リポごと**に持つ（1 本を使い回さない）。turbo リポと base リポでは
#: 焼き込みの有無も格納系列の数も違うので、共有した瞬間にどちらかの告知が事実と食い違う —
#: 値としては妥当な散文なので、`verify_dist` も manifest 検査も素通りして配ってから露見する。
#: MUST: 文面は分割の有無に依らず正しいものを 1 本だけ持つ（manifest を見て出し分けない）—
#: `root_files` は Pipeline 構築時に固定で組むので、manifest 依存の分岐はここに置けない。
#: 「1 個の safetensors ファイル」と綴っていた旧文面は 2026-08-29 の shard 分割で事実と食い違い、
#: §3(d)(i) が求める「改変内容の告知」が改変内容を述べていない状態になっていた（X2-103）。
CONTAINER_MODIFICATION = (
    "- The weights were converted into the container format of the WebGPU inference"
    " runtime Karume\n"
    "  (safetensors holding the weights plus an inference graph in `__metadata__`,"
    " split across\n"
    "  numbered shards when a component is too large for one file)."
)


def notice_markdown(modifications: tuple[str, ...]) -> str:
    """配布リポ直下の `NOTICE.md`。

    §3(b)（Attribution Notice の掲示）+ §3(d)(i)（改変した旨を **Attribution Notice の中に**
    含める）+ §3(d)(iii)（公式製品と誤認させない）を 1 枚で満たす。逐語ブロックは
    {@link ATTRIBUTION_NOTICE}（`anima/card.py` が正本）で、残りは Karume 側の事実の記述。
    改変記載を独立節にせず Notice 節の内側へ置くのは §3(d)(i) の文言（"include in the
    Attribution Notice"）に厳格に合わせるため。
    """
    return (
        "\n".join(
            [
                "# Notice",
                "",
                "## Attribution Notice",
                "",
                ATTRIBUTION_NOTICE,
                "",
                "As required by the license, this Attribution Notice also states that the",
                "applicable CircleStone Model has been modified: this distribution is a Derivative",
                "of the CircleStone Anima model, modified as follows:",
                "",
                *modifications,
                "",
                "## Not an official product",
                "",
                "This is not an official product of CircleStone Labs LLC, and it is not endorsed,",
                "approved or validated by CircleStone Labs LLC.",
                "",
                "The full license text is distributed alongside this repository as LICENSE.md.",
            ]
        )
        + "\n"
    )


#: 公式リポ（karume-anima — CircleStone の 5 変種同居）の告知。
#: 旧 base 告知の「int4 series を足した」は i4 席の無い現物と食い違っていた（2026-09-01 に
#: 是正 — 全モデル i4 なしへ揃えた同日裁定で int4 の行自体が消えた。旧 turbo 告知にあった
#: LoRA 焼き込みの行も公式 checkpoint 化で消えた）。
OFFICIAL_NOTICE_MARKDOWN = notice_markdown(
    (
        CONTAINER_MODIFICATION,
        "- An int8-quantized series of the transformer was added alongside the f16 one.",
    )
)

#: 追加学習リポ（karume-anima-extra — 第三者 fine-tune）の告知。出所は README へ委ねる形で
#: 書く（上の NOTE と同じ理由）。
EXTRA_NOTICE_MARKDOWN = notice_markdown(
    (
        CONTAINER_MODIFICATION,
        "- An int8-quantized series of the transformer was added alongside the f16 one.",
        "- Every model in this repository is a community fine-tune of the CircleStone",
        "  Anima base model; its origin, author and license terms are stated in README.md.",
    )
)

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
#: 役割名でだけ引くので、綴りが 2 箇所で独立に動くことは起きない。
OUTPUT_PATHS: Mapping[str, str] = {
    "text_encoder": "text_encoder/model.safetensors",
    "text_conditioner": "text_conditioner/model.safetensors",
    "transformer_f16": "transformer/model.f16.safetensors",
    "transformer_i8": "transformer/model.i8.safetensors",
    "transformer_i4": "transformer/model.i4.safetensors",
    "rope_base": "transformer/rope_base.safetensors",
    "vae_decoder": "vae_decoder/model.safetensors",
    "tokenizer": "tokenizer/qwen2-tokenizer.json",
    "tokenizer_2": "tokenizer_2/t5-tokenizer.json",
}

#: 席名の部品上書きトークン → その weights 名（ADR 0074 決定 4 — **略称の定義は recipe が
#: 持ち、生成モデルカードの quant 表に対応を必ず出す**）。Anima の基底格納は `f16`（text 経路
#: 3 役が f16 固定）で、圧縮が掛かるのは transformer だけなので席名は `f16+dit<ビット>` になる。
ANIMA_QUANT_ABBREVIATIONS: Mapping[str, str] = {"dit": "transformer"}

#: quant 表（v1 の `presets` — ADR 0041 §3 で改名）。`session` の語彙は manifest 所有。
#: 席名は ADR 0074 の文法 `<格納>[+<部品><ビット>]…[-<ノブ>]…` で、`<格納>` は**全役割に共通の
#: 基底格納**（Anima は f16）。**dtype ラベルが 1 つしかない weights は書かない** —
#: {@link complete_quant_weights} が完全写像へ埋める（写せば済む席を quant の数だけ複製しない）。
#:
#: `label` / `description` は選択 UI 向けの表示欄（ADR 0075 決定 1 — 英語・64 / 200 字上限）。
#: 席の実態（格納・ノブ・嬉しさ）から書く: id は機械の都合で、人が読むのはこの 2 欄という
#: 役割分担を取る。既定であることは書かない（`defaultQuant` が既に指している — ADR 0075 決定 3）。
ANIMA_QUANTS: Mapping[str, Any] = {
    "f16": {
        "weights": {"transformer": "f16"},
        "session": {},
        "label": "Full quality (f16)",
        "description": "Transformer in f16 storage with f32 compute — the largest download,"
        " and the reference the other quants here are judged against.",
    },
    "f16+dit8": {
        "weights": {"transformer": "i8"},
        "session": {},
        "label": "Half size (int8 transformer)",
        "description": "Transformer stored as int8 and computed in f32: roughly half its f16"
        " download, with the execution path left unchanged.",
    },
    "f16+dit8-a8": {
        "weights": {"transformer": "i8"},
        "session": {"linearCompute": "a8"},
        "label": "Half size, int8 linear",
        "description": "The int8 transformer with per-token int8 activations in its linear"
        " layers — faster on GPUs with dp4a, same download.",
    },
    "f16+dit8-a8-attn8": {
        "weights": {"transformer": "i8"},
        "session": {"linearCompute": "a8", "attentionCompute": "a8"},
        "label": "Half size, int8 linear and attention",
        "description": "Adds int8 activations inside attention on top of the int8 linear path;"
        " same weights, one more integer stage per step.",
    },
    "f16+dit8-a8-attn8-s16": {
        "weights": {"transformer": "i8"},
        "session": {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        },
        "label": "Balanced (int8)",
        "description": "The int8 linear and attention path with attention scores held in f16 —"
        " the fastest of the int8 seats here, at f16-level image quality.",
    },
    # i4 常駐の 2 席（波 J-4a — 低 VRAM 席）。
    #
    # MUST: **`linearCompute` を宣言しない**。**理由は 2026-08-21 に入れ替わった**ので注意 —
    # 以前は「a8 の述語が i8 常駐を必要条件に含むので宣言しても 1 バイトも変わらない嘘の席に
    # なる」だったが、w4a8 の実装（ADR 0076）で i4 常駐も整数内積の経路に乗るようになった。
    # 今の理由は**品質**: 宣言すると linear の活性が per-token i8 になり、実 GPU の画で
    # 「細部に破綻・線がラフ」というユーザー視認裁定が出た（2026-08-21・研究記録
    # `docs/research/2026-08-21-anima-i4-seat-speed.md` §6）。速度は 1,640 → 955 ms/step と
    # 大きく戻るが、この席の存在理由は**サイズと VRAM** であって速度ではない（速度が要るなら
    # 既定の `f16+dit8-a8-attn8-s16` が 823 ms/step で上）。attention 側の 2 つは重みスロットを
    # 見ないので i4 常駐でもそのまま効き、視認でも劣化は出ていないので宣言する。
    "f16+dit4": {
        "weights": {"transformer": "i4"},
        "session": {},
        "label": "Smallest (int4 transformer)",
        "description": "Transformer weights in GPTQ-calibrated int4 (group-32) with f32 compute —"
        " the smallest download and the least resident memory.",
    },
    "f16+dit4-attn8-s16": {
        "weights": {"transformer": "i4"},
        "session": {"attentionCompute": "a8", "attentionScoreStorage": "f16"},
        "label": "Smallest, int8 attention",
        "description": "The int4 transformer with int8 activations and f16 scores in attention:"
        " the low-memory seat, without slowing down as much as plain int4.",
    },
    "f16-c16": {
        "weights": {"transformer": "f16"},
        "session": {"linearCompute": "f16", "attentionCompute": "f16"},
        "gpuFeatures": {"shaderF16": True},
        "label": "Full quality, f16 compute",
        "description": "f16 storage computed in f16 throughout. Needs the shader-f16 GPU feature,"
        " and trades numerical headroom for speed.",
    },
}

ANIMA_DEFAULT_QUANT = "f16+dit8-a8-attn8-s16"

#: パイプライン所有の設定（hub は素通し — ADR 0041 §2）。値は移行元の実装定数と一致する:
#: `shift` / `numTrainTimesteps` は sampler の `ANIMA_SHIFT` / `ANIMA_NUM_TRAIN_TIMESTEPS`
#: （エクスポータ側の `SHIFT` / `NUM_TRAIN_TIMESTEPS` = scheduler_config.json と同値）、
#: `steps` / `guidanceScale` は turbo 既定（8 step / cfg 1 — ADR 0038 Examples が正。品質目視
#: ゲート・最終ベンチ・PNG 参照 sha の採取は全て 8 step で行われており、配布既定はそれに揃える。
#: 移行元 CLI の 10 は検証履歴を持たない値）。`negativePrompt` は既定ネガティブプロンプト。
#: `resolution` だけは移行元 CLI の既定（512）を採らない — あちらの 512 は「静的資産の最小」
#: であって推奨値ではなく、配布形は S 形 1 本（ADR 0038 §4）で解像度に依存しない。配布の
#: 推奨既定は ADR 0038 Examples のとおり 1024²。
#: `type`（サンプラ種別）は **Euler を配布既定に維持する**（再裁定 2026-08-25 — 上流 Anima の
#: 推奨サンプラーに合わせる・ユーザー方針「DPM++ 2M は選択肢の一つ」）。0.5.0 で一度
#: dpmpp-2m を宣言したが同日戻した。視認 A/B の観測（同 step 数で dpmpp-2m が同等以上 —
#: seed 42 ほぼ互角・seed 7 は構図と中景の解像で優位・破綻の追加なし）は観測として有効なまま
#: で、既定の裁定だけが上流推奨側に立つ。DPM++ 2M は request 側の `sampler` 席（0.5.1）で
#: 選ぶ。省略時既定も "euler"（models 側 config.ts）だが、裁定済みであることが読めるよう
#: 明示宣言する。
ANIMA_SCHEDULER: Mapping[str, Any] = {"type": "euler", "shift": 3, "numTrainTimesteps": 1000}

#: 既定のネガティブプロンプト。turbo 席では**使われない**（CFG=1 なので uncond 側を 1 度も
#: 計算しない）が、欄自体は据え置く — 利用者が guidance を上げた瞬間に効く値だから。
#: CFG 運用の素版では i4 の**校正入力**の uncond 分岐にもこれが通る（`anima.calib`）。
ANIMA_NEGATIVE_PROMPT = "low quality, worst quality, blurry, bad anatomy, jpeg artifacts"

#: 配布の推奨解像度（ADR 0038 Examples）。移行元 CLI の 512 は「静的資産の最小」であって
#: 推奨値ではなく、配布形は S 形 1 本（ADR 0038 §4）で解像度に依存しない。
ANIMA_RESOLUTION: Mapping[str, int] = {"width": 1024, "height": 1024}

ANIMA_TURBO_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": ANIMA_SCHEDULER,
    "defaults": {
        "steps": 8,
        "guidanceScale": 1,
        "resolution": ANIMA_RESOLUTION,
        "negativePrompt": ANIMA_NEGATIVE_PROMPT,
    },
}

#: 素の base 系の既定。turbo と違い **CFG を使う**（= negative prompt が効く）ので、1 step が
#: forward 2 本になり、step 数も蒸留前の相場へ戻る。
#:
#: step 数は 20〜32 を 3 プロンプト × 4 段で焼いて視認裁定した結果の **20**（2026-08-22）。
#: 既定は「品質に問題の無い範囲で最速」を採る方針で、20 と 32 の差は題材によらず小さかった
#: （step 数を変えると sigma 列ごと変わるので**同じ絵が精細になるのではなく別の絵になる** —
#: 「どこから破綻が減って安定するか」で見る）。上げたい利用者は `steps` を渡せばよい。
ANIMA_BASE_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": ANIMA_SCHEDULER,
    "defaults": {
        "steps": 20,
        "guidanceScale": 4,
        "resolution": ANIMA_RESOLUTION,
        "negativePrompt": ANIMA_NEGATIVE_PROMPT,
    },
}

#: 公式 Aesthetic 変種の既定。CFG を使う点は base と同じ。
#:
#: step 数は 20 / 30 / 50 を seed 42-45 × 4 題材（人物・風景・遠近・全身+街）で焼いて視認
#: 裁定した **30**（2026-09-01・v1.0 / v1.1 両方で確認）。50 は良くなっている気はするが
#: 20 からの劇的改善ではない、の中庸 — 上流 README の一般推奨（30-50 step / CFG 4-5）の
#: 下端でもある。上げたい利用者は `steps` を渡せばよい。
ANIMA_AESTHETIC_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": ANIMA_SCHEDULER,
    "defaults": {
        "steps": 30,
        "guidanceScale": 4,
        "resolution": ANIMA_RESOLUTION,
        "negativePrompt": ANIMA_NEGATIVE_PROMPT,
    },
}

#: 各役割の safetensors ヘッダに**要求する格納 dtype**（存在検査）。実測の事故が根拠:
#: f16 系列のつもりで `--dtype` を付け忘れた素の F32 資産は、組み立て・ロード・実行の全てを
#: 通って**PNG の参照一致まで露見しなかった**。格納形は series ディレクトリ名でなくヘッダが正。
#: f16 系列は fake-quant 対象だけが F16 になる（norm/bias 等は F32 のまま）ので「F16 を含む」
#: を要求する。rope_base（F32 のみ）と tokenizer（JSON）はここに載せない。
#: i4 系列は**混成**（F32 + I8 + I4 が同居する）なので **I4 を要求する** — {@link assert_storage}
#: は「要求 dtype がヘッダに在る」を見るので、I8 を要求すると i8 系列が i4 席へ入っても素通りし、
#: 席の取り違えが沈黙する（sbv2 の i4 席と同じ規律）。
STORAGE_REQUIREMENTS: Mapping[str, str] = {
    "text_encoder": "F16",
    "text_conditioner": "F16",
    "transformer_f16": "F16",
    "transformer_i8": "I8",
    "transformer_i4": "I4",
    "vae_decoder": "F16",
}

#: 各役割の safetensors ヘッダに**あってはならない**格納 dtype（{@link assert_storage_absent}）。
#: 存在検査だけでは**圧縮席どうしの取り違え**が素通りする — i4 系列は混成で、既定格納が i8
#: （`anima/export.py` の `BASE_WEIGHT_DTYPES`）なので **必ず I8 を含む**。したがって i4 系列を
#: `transformer_i8` へ挿し込む取り違えは「I8 を含む」を満たしてしまい、組み立ても verify_dist も
#: ロードも通る。実害は既定 quant `f16+dit8-a8-attn8-s16` に出る: 宣言した `linearCompute: "a8"`
#: の述語は `c285f97` 以降 i4 常駐も受ける（ADR 0076）ので、常駐が i4 だと fail loudly せず
#: **w4a8 の数値契約**（group 部分縮約）で走る — ADR 0076 決定 6 が「画が荒れるので席に載せない」
#: と決めた構成が、席名が int8 を名乗ったまま既定席で沈黙して出る。
#: MUST: 禁止は**役割ごとに集合**で持つ（1 つだけだと 4 本目の系列が生えた日に、名指ししなかった
#: ほうが黙って素通りする — irodori と同じ規律）。f16 席は I8 / I4 の不在で二重に締まる。
ANIMA_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    "transformer_f16": ("I8", "I4"),
    "transformer_i8": ("I4",),
}

#: weights の宣言（dtype ラベル → 役割名）。ラベルは**格納 dtype 語彙**で、
#: {@link STORAGE_REQUIREMENTS} が要求する格納形と 1:1（ADR 0041 §3）。`i4` は**混成の系列**を
#: 指すラベルで、実体は「i4 適格な重みが i4 group32・残りは i8」（`anima/export.py`）。
ANIMA_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    "text_encoder": {"f16": WeightFiles("text_encoder")},
    "text_conditioner": {"f16": WeightFiles("text_conditioner")},
    "transformer": {
        "f16": WeightFiles("transformer_f16", {"rope_base": "rope_base"}),
        "i8": WeightFiles("transformer_i8", {"rope_base": "rope_base"}),
        "i4": WeightFiles("transformer_i4", {"rope_base": "rope_base"}),
    },
    "vae_decoder": {"f16": WeightFiles("vae_decoder")},
}

#: assets の宣言（quant 選択に依存しない無条件ファイル — ADR 0041 §3）。
ANIMA_ASSETS: Mapping[str, str] = {"tokenizer": "tokenizer", "tokenizer_2": "tokenizer_2"}


@dataclass(frozen=True)
class AnimaModel:
    """モデル 1 つぶんの Anima 固有の事実（系列の引き方・席の範囲・既定値の違い）。

    MUST: **LoRA の有無を bool ではなく sha256 で持つ**。「焼いていない」と「焼いたが記録が
    無い」は資産からは区別できないので、期待値を持たない側（`None`）は**記録が無いこと**を
    積極的に検査する（{@link assert_lora_provenance}）。片方向の検査だけだと、turbo の系列を
    素モデルの席へ挿し込む取り違えが素通りする。
    """

    #: 焼き込んだ LoRA の sha256。`None` = 素（焼いていない）。
    lora_sha256: str | None
    #: transformer の格納 dtype ラベル。ここに無い格納形の系列は**要求も宣言もしない**
    #: （2026-08-23・波 J-4 ②: 校正条件を {@link pipeline_config} から導く形にした
    #: 〈`anima.calib.calib_conditions`〉ので、i4 席は turbo 専用ではなくなった）。
    storages: tuple[str, ...]
    #: text_conditioner を自前で持つか。`False` = 素の共有系列（{@link ANIMA_BASE_SERIES}）。
    #: 第三者 fine-tune は DiT だけでなく llm_adapter（= text_conditioner）も焼き直している
    #: ので、共有すると**別のモデルのテキスト条件付け**で走る（絵だけが静かにずれる）。
    own_text_conditioner: bool
    #: manifest の `pipelineConfig`（step / guidance がモデルごとに違う）。**i4 の校正条件も
    #: ここから導く**（`anima.calib.calib_conditions`）— 校正が見る sigma 列と CFG 分岐を配布
    #: 実行時の条件そのものにするため、export 側に写しを置かない。ここを動かすと次の i4
    #: export の丸め先も動く。
    pipeline_config: Mapping[str, Any]
    #: i4 系列に要求する丸め方式（`calib_provenance.json` の `method`）。配布は 1 通りだけ
    #: （{@link CALIB_SHIPPABLE_METHOD}）で、**動かすのは視認評価専用の席だけ**
    #: （`anima.eval_dist` — 量子化感度の実験変種を手元で組んで見るための spec）。既定を
    #: 定数のままにしてあるので、配布経路の 4 モデルはこの欄を 1 度も綴らない。
    calib_method: str = CALIB_SHIPPABLE_METHOD


#: モデル名 → 事実。リポの分かれ目でもある（{@link OFFICIAL_MODELS} / {@link EXTRA_MODELS}）。
#:
#: NOTE: 全モデルが **"i4" を持たない**（席とファイルが {@link anima_quants} /
#: {@link anima_weights} の導出で両方消える）— i4 の視認裁定（2026-08-24）で構図分岐が大きく
#: 配布スキップ、0.5.0 の上げ直し（2026-08-25）でも除外の裁定。旧 fused turbo だけが持って
#: いた i4 席も公式 checkpoint 化（ADR 0087）の際に持ち越さない裁定（2026-09-01）。復活条件 =
#: adaLN 関連で出ていた量子化感度の高い部分の特定（旧系列は outputs/series/ の *-i4-dyn に
#: 温存 — 校正済み退避の series-archive は 2026-08-30 の掃除裁定で削除済み）。
ANIMA_MODELS: Mapping[str, AnimaModel] = {
    # 公式 Turbo（蒸留済み checkpoint そのもの — 旧 anima-turbo の LoRA 焼き込みは不要に
    # なった）。lora_sha256=None は「焼いていない」の積極検査で、lora_provenance.json を持つ
    # 旧 fused 系列の挿し込みを落とす。text_conditioner は base と f16 丸め後ビット同一の実測
    # （2026-09-01）で共有系列を使う。旧 fused turbo が持っていた i4 席は**持ち越さない**
    # （2026-09-01 ユーザー裁定 — 全モデル i4 なしに揃う。復活レバーは素版と同じ
    # `anima.eval_dist`）。
    ANIMA_TURBO_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8"),
        own_text_conditioner=False,
        pipeline_config=ANIMA_TURBO_PIPELINE_CONFIG,
    ),
    ANIMA_BASE_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8"),
        own_text_conditioner=False,
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
    # 公式 Aesthetic。i4 席を持たないのは素版と同じ裁定線（上の NOTE — 全モデル i4 なし）。
    ANIMA_AESTHETIC_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_AESTHETIC_PIPELINE_CONFIG,
    ),
    # v1.0 世代の公式 2 変種も並行配布（2026-09-01 ユーザー裁定 — バージョン間で好みが
    # 分かれる系なので旧版へ戻る先を残す = ADR 0077 の動機そのもの）。conditioner は
    # どちらも base と f16 丸め後 64/118 テンソル差の実測（2026-09-01）で自前。
    "anima-turbo-v1.0": AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_TURBO_PIPELINE_CONFIG,
    ),
    "anima-aesthetic-v1.0": AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_AESTHETIC_PIPELINE_CONFIG,
    ),
    "anima-wai-v1.0": AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
    "anima-copycat-20260610": AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
}

#: リポごとの受理集合。**Pipeline が違えば直下の法的テキスト（NOTICE）も違う**ので、
#: 取り違えて組むと「改変告知が中身と食い違うリポ」が黙って出来上がる — 計画段で落とす。
#: 分割の軸は**公式 / 追加学習**（2026-09-01 裁定 — ADR 0087）: 公式リポ（karume-anima）は
#: CircleStone の 5 変種同居・既定 = Turbo、追加学習リポ（karume-anima-extra）は第三者
#: fine-tune で text stack を公式リポへ越境参照する。
OFFICIAL_MODELS: tuple[str, ...] = (
    ANIMA_TURBO_MODEL_NAME,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_AESTHETIC_MODEL_NAME,
    "anima-turbo-v1.0",
    "anima-aesthetic-v1.0",
)
EXTRA_MODELS: tuple[str, ...] = ("anima-wai-v1.0", "anima-copycat-20260610")


@dataclass(frozen=True)
class AnimaSources:
    """組み立ての入力となる系列ディレクトリ群。

    テキスト経路と VAE は DiT の格納 dtype に依らないので f16 系列 1 本を共有する
    （ADR 0019）。transformer だけが f16 / i8 / i4 の 3 系列に分かれる。
    """

    #: 格納 dtype ラベル → transformer 系列（{@link AnimaModel.storages} の順）。
    transformer: Mapping[str, Path]
    base: Path
    #: text_conditioner を持つ系列（素は {@link base} と同じ）。
    text_conditioner: Path
    tokenizers: Path


def anima_sources(series_dir: Path, model: str = ANIMA_BASE_MODEL_NAME) -> AnimaSources:
    """系列の親ディレクトリ（`outputs/series/`）から 1 モデルぶんの系列を引く。

    モデル名は transformer の系列と、自前の text_conditioner を持つモデルの静的系列にだけ
    掛かる — text_encoder / VAE / tokenizer は上流の fine-tune が触らない部分なので、
    どのモデルでも共有系列 1 本から引く。
    """
    spec = anima_model(model)
    return AnimaSources(
        transformer={storage: series_dir / f"{model}-{storage}-dyn" for storage in spec.storages},
        base=series_dir / ANIMA_BASE_SERIES,
        text_conditioner=series_dir
        / (f"{model}-f16" if spec.own_text_conditioner else ANIMA_BASE_SERIES),
        tokenizers=series_dir / ANIMA_TOKENIZER_SERIES / "text",
    )


def anima_model(model: str) -> AnimaModel:
    """モデル名から事実を引く（知らない名前は選択肢を並べて落とす）。"""
    spec = ANIMA_MODELS.get(model)
    if spec is None:
        choices = ", ".join(ANIMA_MODELS)
        raise DistError(f"知らない Anima のモデル名: {model!r}（選択肢: {choices}）")
    return spec


def transformer_series(sources: AnimaSources) -> tuple[Path, ...]:
    """格納 dtype 別の transformer 系列（**系列横断の突合はこの 1 本から引く** MUST）。

    rope 素表のバイト同一検査と LoRA 帰属の突合はどちらも「全系列を舐める」検査で、列挙を
    2 箇所に持つと格納席が増えた日に片方だけ更新される — 網から漏れた系列は検査を素通りし、
    どちらの綻びも実行時には沈黙する（幾何違いは絵だけ壊れ、帰属違いは README だけが嘘になる）。
    """
    return tuple(sources.transformer.values())


def shared_rope_base(sources: AnimaSources) -> Path:
    """全 transformer 系列の rope 素表がバイト同一であることを確かめ、1 本化する元を返す。

    MUST: `rope_base.safetensors` は f16 / i8 / i4 の各系列に同名で並ぶ。全てのバイト同一を
    sha256 で確かめてから 1 本化する — 食い違ったまま 1 つを選ぶと、選ばれなかった系列の
    quant が「別の幾何の rope 表で走る」形になり、ロードも実行も通って絵だけが静かに壊れる。
    """
    candidates = [
        series / "transformer" / "rope_base.safetensors" for series in transformer_series(sources)
    ]
    for path in candidates:
        if not path.is_file():
            raise DistError(f"組み立ての入力が無い: {path}")
    digests = {path: sha256_file(path) for path in candidates}
    if len(set(digests.values())) != 1:
        listing = "\n".join(f"  {digest}  {path}" for path, digest in digests.items())
        raise DistError(
            "rope_base.safetensors が系列間でバイト同一でない — 1 本化できない。"
            f"どちらが正かはここでは決められないので組み立てを止める:\n{listing}"
        )
    return candidates[0]


def assert_lora_provenance(sources: AnimaSources, expected: str | None) -> None:
    """transformer 系列が記録した LoRA の sha256 が、モデルの宣言と一致することを確かめる。

    MUST: モデルの宣言（{@link AnimaModel} の `lora_sha256`）が「どの LoRA を焼いた配布物か」の
    唯一の記録なのに、融合後の重みからは焼いた LoRA を復元できない。突き合わせが無いと、
    LoRA を差し替えて再エクスポートしても古い / 誤った sha256 のまま組み上がる — 値は 64 桁
    hex として形式が妥当なので `verify_dist` の構造検査も通り、**沈黙する**。「別々の台本が持つ同じ
    事実は組み立て時に必ず突き合わせる」（rope_base のバイト同一検査と同じ規律）。

    MUST: `expected is None`（素のモデル）では**記録が無いことを検査する**。融合済みの重みと
    素の重みは資産の形が 1 バイトも変わらないので、turbo の系列を素モデルの席へ挿し込む
    取り違えは他のどの検査にも掛からない。書き手（`anima/export.py`）は LoRA を焼かなかった
    ターゲットの記録を消す（全域関数）ので、**記録の不在が「焼いていない」の証跡**になる。
    """
    for series in transformer_series(sources):
        path = series / "transformer" / LORA_PROVENANCE_FILE
        if expected is None:
            if path.is_file():
                raise DistError(
                    f"LoRA を焼いていないモデルの席に、焼いた記録のある系列が来ている: {path}"
                    "（`--lora` を付けずに再エクスポートすると記録は消える）"
                )
            continue
        if not path.is_file():
            raise DistError(
                f"焼き込んだ LoRA の記録が無い: {path}"
                "（`python -m anima.export --lora …` で再エクスポートすると書かれる）"
            )
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as cause:
            raise DistError(f"LoRA の記録を解析できない: {path} — {cause}") from cause
        digest = record.get("sha256") if isinstance(record, dict) else None
        if digest != expected:
            raise DistError(
                f"焼き込んだ LoRA がカードの宣言と違う: {path} は {digest!r}、"
                f"anima/card.py は {expected!r} — どちらが正かはここでは決められない"
            )


def assert_calib_provenance(sources: AnimaSources, spec: AnimaModel) -> None:
    """i4 系列が**このモデルの条件で・配布して良い校正**（GPTQ × 下限以上の予算 × モデル別の
    step / CFG）で丸められたことを、書き出し側の記録で確かめる。

    MUST: 校正の方式も予算も条件も格納形を 1 バイトも変えない
    （`research/2026-08-21-anima-i4-seat-speed.md` §6 — ファイルサイズは RTN 版とバイト単位で
    同じ）。したがって `verify_dist` の構造検査もヘッダ dtype 検査も**素通りする**。`--no-calib`
    は smoke 用の opt-out なのに、その生成物が配布へ紛れても資産からは判別できず、出るのは
    「全体的にぼやけた」絵だけ — LoRA 帰属（{@link assert_lora_provenance}）と同じ「別々の台本が
    持つ同じ事実は組み立て時に必ず突き合わせる」規律をここにも敷く。

    MUST: **`spec` を受け取って条件まで突き合わせる**（要求する丸め方式も {@link
    AnimaModel.calib_method} から引く — 配布は 1 通りで、動くのは視認評価専用の席だけ）。
    `anima.export` の `--model` は校正条件を
    引くためだけのノブ（LoRA 焼き込みのような格納バイトへの影響が無い）なので、素版の重みを
    `--model anima-turbo` で焼いた資産は「正しい素版 i4」に見える — 校正だけが turbo の
    8 step・CFG 1 で回っている。条件はモデルの `pipeline_config` から導かれる
    （`anima.calib.calib_conditions`）ので、同じ 1 箇所から引き直せば突き合わせられる。

    判定そのものは `_shared.calib_provenance` が正本（irodori の 2 つの読み手と同じ 1 実装）。
    """
    from .calib_prompts import DEFAULT_CALIB_PROMPTS

    path = sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE
    if not path.is_file():
        raise DistError(
            f"i4 系列の校正条件の記録が無い: {path}"
            "（`python -m anima.export --dtype i4` で再エクスポートすると書かれる）"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as cause:
        raise DistError(f"校正条件の記録を解析できない: {path} — {cause}") from cause
    defaults = spec.pipeline_config["defaults"]
    complaint = calib_complaint(
        record,
        method=spec.calib_method,
        at_least={"prompts": DEFAULT_CALIB_PROMPTS},
        exactly={"steps": int(defaults["steps"]), "guidance": float(defaults["guidanceScale"])},
    )
    if complaint is not None:
        raise DistError(
            f"i4 系列の校正条件が要求と食い違う: {path} は{complaint}"
            " — `--no-calib` / smoke 予算 / 別モデルの条件 / 役割別の実験変種で焼いた生成物は"
            "配布に使わない"
        )


def anima_placements(sources: AnimaSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（`io.*.safetensors` を落とす仕掛けはこれで足りる）。
    """
    placements = {
        "text_encoder": sources.base / "text_encoder" / "model.safetensors",
        "text_conditioner": sources.text_conditioner / "text_conditioner" / "model.safetensors",
        "rope_base": shared_rope_base(sources),
        "vae_decoder": sources.base / "vae_decoder" / "model.safetensors",
        "tokenizer": sources.tokenizers / "qwen2-tokenizer.json",
        "tokenizer_2": sources.tokenizers / "t5-tokenizer.json",
    }
    for storage, series in sources.transformer.items():
        placements[f"transformer_{storage}"] = series / "transformer" / "model.safetensors"
    return placements


def anima_weights(spec: AnimaModel) -> Mapping[str, Mapping[str, WeightFiles]]:
    """モデルが持つ格納形だけへ絞った weights 宣言（席の無い dtype ラベルは載せない）。"""
    return {
        role: (
            {label: files for label, files in labels.items() if label in spec.storages}
            if role == "transformer"
            else labels
        )
        for role, labels in ANIMA_WEIGHTS.items()
    }


def anima_quants(spec: AnimaModel) -> Mapping[str, Any]:
    """モデルが持つ格納形で成立する quant 席だけへ絞る。

    MUST: 席の取捨は**宣言した格納形から導く**（席名の直書きリストを別に持たない）。2 箇所に
    分けて持つと、格納形を増やした日に席だけが増えない / 減らした日に席だけが残る。
    """
    return {
        name: entry
        for name, entry in ANIMA_QUANTS.items()
        if entry["weights"]["transformer"] in spec.storages
    }


def anima_plan(
    sources: AnimaSources,
    model: str = ANIMA_BASE_MODEL_NAME,
    *,
    spec: AnimaModel | None = None,
) -> ModelPlan:
    """Anima 1 モデルぶんの計画を組む（検査はここで全部済ませる — 1 バイトも書かない）。

    `spec` は**明示で差し替える席**（省略時はモデル名から引く = 配布経路）。差し替えるのは
    視認評価専用の組み立て（`anima.eval_dist`）だけで、そちらは「素版に i4 席を戻し、要求する
    丸め方式を実験変種の綴りにした」spec を渡す — 配布の受理集合（{@link ANIMA_MODELS}）を
    実験のために動かさないための口で、逆向き（配布経路が spec を綴る）には使わない。
    """
    assert_model_name(model)
    spec = anima_model(model) if spec is None else spec
    assert_lora_provenance(sources, spec.lora_sha256)
    if "i4" in spec.storages:
        assert_calib_provenance(sources, spec)
    placements = anima_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, ANIMA_STORAGE_FORBIDDEN)
    weights = anima_weights(spec)
    return ModelPlan(
        name=model,
        pipeline=ANIMA_PIPELINE,
        artifacts={
            role: Artifact(OUTPUT_PATHS[role], source=source) for role, source in placements.items()
        },
        weights=weights,
        assets=ANIMA_ASSETS,
        quants=complete_quant_weights(weights, anima_quants(spec)),
        default_quant=ANIMA_DEFAULT_QUANT,
        pipeline_config=spec.pipeline_config,
    )


def anima_dist_plan(series_dir: Path, model: str, allowed: tuple[str, ...]) -> ModelPlan:
    """`--series` の親から Anima 1 モデルの計画を組む（CLI のディスパッチ先）。

    MUST: `allowed` は**その Pipeline のリポに入ってよいモデル**。Pipeline が違えばリポ直下の
    NOTICE（改変告知）も違うので、取り違えて組むと「告知が中身と食い違うリポ」が黙って
    出来上がる — manifest も verify_dist も構造としては正しいままなので、配ってからでないと
    誰も気づけない。
    """
    if model not in allowed:
        choices = ", ".join(allowed)
        raise DistError(
            f"モデル {model!r} はこの pipeline のリポに入らない（入るのは: {choices}）— "
            "リポ直下の改変告知が中身と食い違うので組み立てを止める"
        )
    return anima_plan(anima_sources(series_dir, model), model)


def root_files(notice: str) -> dict[str, str]:
    """配布リポ直下へ入れる法的テキスト（`karume.dist.Pipeline.root_files`）。

    ライセンス原文は recipe に置いた現物（{@link LICENSE_SOURCE_PATH}）を**逐語で**読む —
    ここで整形や差し替えをすると §3(a) の「このライセンスのコピー」ではなくなる。改変告知
    （`NOTICE.md`）だけがリポごとに違うので、そこだけ引数で受ける。
    """
    return {
        "LICENSE.md": LICENSE_SOURCE_PATH.read_text(encoding="utf-8"),
        "NOTICE.md": notice,
    }


#: 公式リポの配布名（`karume-` prefix はリポ名裁定 2026-08-09 — HF org の代わりの名前空間）。
OFFICIAL_REPO_NAME = "karume-anima"

#: 追加学習リポの配布名。
EXTRA_REPO_NAME = "karume-anima-extra"


def _official_repo_name(_model: str) -> str:
    """公式リポ名（ADR 0087 で 5 変種が 1 リポへ畳まれたのでモデル名を見ない）。

    `Pipeline.repo_name` は「単一モデルを組んだときの既定の出力先」を答える席で、ここが
    `karume-<モデル名>` を返すと `--out` 省略時の出力先とカードの Usage 例が実在しない
    リポ名になる。
    """
    return OFFICIAL_REPO_NAME


def _extra_repo_name(_model: str) -> str:
    """追加学習リポ名（公式と同じく 1 リポ複数モデル）。"""
    return EXTRA_REPO_NAME


#: `--pipeline anima` の 1 行（公式リポ karume-anima — CircleStone の 5 変種同居・既定 =
#: Turbo〈上流 README の推奨・2026-09-01 裁定〉）。
#:
#: MUST: extra と**別の Pipeline**にする。`root_files` は Pipeline に固定で載る 1 組なので、
#: 1 つに畳むと改変告知がどちらかのリポで嘘になる（§3(d)(i) の要件を落とす）。
OFFICIAL_PIPELINE = Pipeline(
    default_model=ANIMA_TURBO_MODEL_NAME,
    repo_name=_official_repo_name,
    plan=lambda series_dir, model: anima_dist_plan(series_dir, model, OFFICIAL_MODELS),
    card_profiles={"anima": partial(render_base_card, abbreviations=ANIMA_QUANT_ABBREVIATIONS)},
    # 上流ライセンスの再配布条件（§3）は配布リポ 1 つに掛かるので、読みも組み立ての回数に
    # よらず**ここで 1 回**。
    root_files=root_files(OFFICIAL_NOTICE_MARKDOWN),
)

#: `--pipeline anima-extra` の 1 行（追加学習リポ karume-anima-extra — 第三者 fine-tune。
#: text stack は公式リポへ越境参照して組む: dist の `--ref-*` 5 指定・runbook の公開順序）。
EXTRA_PIPELINE = Pipeline(
    default_model="anima-wai-v1.0",
    repo_name=_extra_repo_name,
    plan=lambda series_dir, model: anima_dist_plan(series_dir, model, EXTRA_MODELS),
    card_profiles={
        "anima-extra": partial(render_extra_card, abbreviations=ANIMA_QUANT_ABBREVIATIONS)
    },
    root_files=root_files(EXTRA_NOTICE_MARKDOWN),
)
