# OP 実装の数値レビュー — 正しさ・危険クラス・精度（2026-08-31）

> 性格: **時点スナップショット**（HEAD b35cf5c 時点の横断監査記録）。修正はレビュー直後の
> コミット列で消化済み — 現況は git log とコード側 doc が正。レビューの一次台帳
> （findings / 敵対検証の全文）は `.claude/reviews/2026-08-31_b35cf5c/`（git 追跡外）。

## 経緯と方法

67eb07a（tanh_stable — Metal fast-math の沈黙 NaN 根治）を受けた「数値危険クラス監査波」を、
OP 面全域（kernels 35 / codegen 7 / reference 4 / ops 4 + tests の tolerance 体制）の
フォーカス網羅レビューとして実施。レビュー 10 レッグ（領域別 5 + 横断レンズ 5）→ 敵対検証
5 レッグ（独立再導出・WGSL 仕様原文・torch 実行・配布資産の直読）→ 外部レビュー
（ChatGPT 5 本）の検証 2 レッグ。判定は 24 主題で holds 14 / refuted 8 / uncertain 2。

**結論: C 0 / E 0。** OP 実装の正しさに確定欠陥は無く、指摘は「取り残し 1 本（gru_scan の
素の tanh — 修正済み）・門の欠け・doc の過剰主張・検証体制」に集中した。

## 危険クラス台帳 — 「中間オーバーフロー / アンダーフローだが最終値は有限」

| 経路                              | 守り                                                                                      | 判定                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `tanh` / `gelu_tanh`              | tanh_stable（飽和打ち切り 9.5）                                                           | 安全（下記）                                                            |
| `gru_scan` の候補ゲート tanh      | **無防備だった** → tanh_stable 共有で解消                                                 | 修正済み（実測前活性 max 17.81・層 1 静的上界 31.57 vs NaN 境界 44.36） |
| `sigmoid` / `silu`                | exp の引数を −\|x\| に固定                                                                | 構造的に安全                                                            |
| softmax / attention 統計 / i8a8 P̃ | max 減算（exp(S−m) ≤ 1）                                                                  | 構造的に安全                                                            |
| `softplus`（分解形）              | torch 分解が threshold=20 の where ごと IR に乗る（select は不選択腕の NaN を伝播しない） | 安全 — 出荷 IR の直読で確認                                             |
| `log1p`                           | 級数切替 + 素の log(1+x)                                                                  | 安全（定義域は契約側）                                                  |
| rms/layer_norm の Σx²             | f32 中間の破綻には \|v\| ≳ 1.8e19 が必要（実測活性と 18 桁差）                            | 到達不能                                                                |
| 除算の分母                        | 全経路で構造的に正（scale 床 tiny・行和は safe 変種のみ空行ガード）                       | 安全（例外 = 融合 attention の全 -inf 行 — W-2 として裁定へ）           |
| `sin`（Snake 活性 α·x）           | WGSL の精度保証域 \|x\|≤π の外（実測 \|αx\| max 71.3）だが e2e 実測で誤差の支配項でない   | 記録のみ（optional）                                                    |
| `quantize_rows` の Inf 行         | limitations 記載の documented behavior（非有限性は必ず残る）                              | 契約どおり                                                              |

## tanh 飽和打ち切りの品質影響 — なし（3 系統で確定）

1. **数学**: f32 で tanh(x) が正確に 1.0 へ丸まる境界は 1−tanh(x) < 2^−25 ⇔ x ≥ 9.0109。
   閾値 9.5 はその上 — 非飽和域はビット不変・飽和域も差 0.188 ulp（丸め境界 0.5 ulp の内側）。
   NaN/±Inf/−0.0/非正規化数の総当たりで反例ゼロ。tanh(±Inf)=NaN（素の exp 経由）の穴を
   副次的に塞いだ。
2. **エコシステム前例**（一次ソース確認）: WGSL 仕様 issue **gpuweb#4458**（open）が同一問題を
   提起し打ち切り閾値 f32=9.010913 を提案。**PyTorch MPS は同型クランプ**（GELU tanh 経路
   `clamp(x,−10,10)`・PR #186286 の根本原因記述 = fast::tanh の exp(2x) 近似も一致）。
   TF.js / ORT Web は式変換 `sign(x)·(1−exp(−2|x|))/(1+exp(−2|x|))`（ORT はコード内で #4458 を
   参照）— 安全だが非飽和域のビットが組込 tanh と変わるためビット同一契約の karume には
   不適合。llama.cpp Metal / MLX は `metal::precise::tanh`（2023 年〜）— WGSL に相当機能は
   無く（naga / Tint は tanh を素通し・無防備を実装で確認）採れない。**9.5 は前例帯
   [9.01, 10.0] の中**。
3. **閾値 20 化の検討**: 算術上は両側余裕が同時改善するが、実際の失敗モード（ドライバ除算の
   2.5 ULP 許容）を +0.19 ulp では閉じない — 採らない（optional 棄却）。

## 主な確定事項と裁定（詳細はレビュー台帳）

- **W-1 gru_scan tanh** → tanh_stable 共有で修正（`is_nan_bits` 複写 5 か所の正本化と同時・
  キー v2・飽和域 parity ケース = Metal 回帰門を常設）。
- **W-4 飽和域の厳密門** → tanh=±1.0 / gelu_tanh 正側=x・負側=−0.0 の Object.is 門 + ±Inf。
- **fma 単一丸め契約の過剰主張**（外部レビュー由来）→ WGSL は fma の精度を x*y+z から継承と
  規定するのみ（真の融合は非保証）。i8a8 族の契約文を「融合実装での性質」へ訂正・atol=0 門は
  device 別 conformance として維持（ADR 0076 追記）。
- **NaN 伝播の非対称**（全 NaN 行が safe_softmax で 0 に化ける — WGSL の max は仕様で NaN を
  落とす）→ ADR 0044 の契約外だが limitations の全称文と不整合。文書修正 + nan_max 統一は
  裁定事項。
- **融合 attention の全 -inf 行ガード欠落**（実配布 mask は literal −inf を運ぶ — 資産直読で
  実証・現行資産は全マスク行ゼロで未発火）→ 裁定事項。
- **検証体制**: DEFAULT_TOLERANCE（rtol 1e-3・省略呼び実測 46 か所）は M1 宿題のまま /
  **f64 オラクル参照は「f32 中間オーバーフロー」クラスを構造的に検出できない**（tanh バグが
  掃引を素通りした機序）→ 危険クラスカナリアの型を op-vocabulary へ規約化。
- 軽微修正: i8a8 QK scale の fround 門・embedding i4 group 整除門・assertEps の subnormal
  拒否・参照 eps の fround・Box–Muller u1=0 の退避先・act_quant scale の乗算形（exporter を
  ランタイム鏡像へ）・erf / MAX_K / PvQuant ラップの doc 訂正。
- **refuted の代表**（蒸し返し防止）: f16 mask sentinel 値域外（実 mask は literal −inf）/
  gelu(−Inf)=−0.0 説（torch も NaN — 実行で反証）/ softplus 無防備説 / cast 沈黙分岐
  （ADR 0062 既決）/ 量子化経路の正しさ欠陥（ゼロ件 — clamp(±127) は導出上到達不能）。

## 見送り・実験計画（起票のみ）

- **GPTQ static-groups + act-order**: upstream の `--static-groups` は「act-order + 推論側
  変更なし」を実装で実証済み（一次確認）— karume の不採用理由は dynamic group 前提でのみ
  成立。実験計画（段階導入・既定 off でビット同一・SBV2/gemma4 の 4 点 sweep・評価分離）を
  backlog へ。
- **GPTQ damping sweep**（0.001〜0.1 × 校正量 1x/4x/16x — 0.01 固定の妥当性を実測で封印）。
- **norm の 1/dim ホスト化**（G1-2）: uncertain — 開発機の除算は正しい丸めと 55,605/200,000
  件で割れる実測があり、**変更すると凍結 sha が割れ得る**。実 GPU プローブなしに採らない。
- reduce identity の params 経由 −inf 化（softmax と同型・L-9）・A8 ±1 bin 率メトリクス
  （一般活性では恒常 0 の見込み — 一度きりの実測で足りる）。
