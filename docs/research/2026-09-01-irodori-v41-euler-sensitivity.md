# Irodori v4.1-small: full-loop golden 検証の許容超過と誤差増幅の実測

> NOTE: 時点スナップショット（2026-09-01 実測）。恒久の判定ロジックは
> `tools/export-recipes/irodori/pipeline_ref.py`（`EULER_REFERENCE_ATOL` 周辺と
> `euler_reference_within_sensitivity`）が正本。

## 事象

`irodori.pipeline_ref --dtype f16`（v4.1-small）が emit 前の検証
（ホスト経路 vs 上流 `sample_euler_rf_cfg` の最終 z 差 ≤ 1e-3）で停止:
case.full の実測 **1.33e-2**。f32 は通過（1.08e-4）、v4-small は全 dtype 通過。

## 実測 — 摂動プローブ

初期 noise に +1e-6 を載せ、最終 z の変位 / 1e-6 = **増幅率 amp**（誤差の蓄積・増幅の倍率）を
全構成で測定。`worst / amp` は「1 step あたりの実装差（蓄積前の種）」の推定量。

| model      | dtype | case   | S   | worst       | amp        | worst/amp |
| ---------- | ----- | ------ | --- | ----------- | ---------- | --------- |
| v4-small   | f16   | full   | 161 | 2.12e-4     | 353        | 6.0e-7    |
| v4-small   | f16   | no-ref | 116 | 7.76e-5     | 222        | 3.5e-7    |
| v4-small   | i8    | full   | 161 | 3.35e-4     | 152        | 2.2e-6    |
| v4-small   | i8    | no-ref | 116 | 2.17e-5     | 28         | 7.8e-7    |
| v4.1-small | f16   | full   | 152 | **1.33e-2** | **37,107** | 3.6e-7    |
| v4.1-small | f16   | no-ref | 115 | 3.93e-5     | 172        | 2.3e-7    |
| v4.1-small | f32   | full   | 152 | 1.08e-4     | 6,884      | 1.6e-8    |
| v4.1-small | f32   | no-ref | 115 | 1.85e-4     | 334        | 5.5e-7    |
| v4.1-small | i8    | full   | 152 | 1.75e-4     | 850        | 2.1e-7    |
| v4.1-small | i8    | no-ref | 115 | 8.91e-5     | 180        | 5.0e-7    |

## 結論

- 超過の真因は**実装差の拡大ではなく誤差の蓄積のされ方**。worst/amp（蓄積前の種）は
  全 10 セルで 1.6e-8〜2.2e-6 に収まり、v4.1-f16 は従来水準以下（3.6e-7）。
- v4.1-f16 case.full の軌道だけが摂動を 37,107 倍に増幅する（v4-small f16 の 105 倍）。
  発散は frame 59〜63（t≈2.4s）に局在し、f32 でも同じ領域が最大 — 内容依存の感度領域。
  f16 の重み丸めが軌道をわずかに動かした先が偶然そこを通った（i8 丸めでは 850 倍で済む）。
- 旧許容 1e-3 の根拠は「v4-small f32 実測 ~1e-4 の約 10 倍 + 式の取り違えは O(1) で出る
  という分離」（当時のコメントに「丸めが条件数の悪い領域を踏めば桁が変わりうる・実測で
  決着させる」と留保済み — 今回その留保が現実化）。

## 音声への影響（実測）

両経路の z を DACVAE decoder（f32）で復号して比較
（`outputs/bench/karume-irodori-v4.1-small/2026-09-01_euler-sensitivity-ab/`）:

- case.full: 波形 SNR **60.0 dB**・最大サンプル差 0.0057（ピーク振幅 0.89 の 0.6%）・
  差のピーク t=2.48s（latent の予測どおり）。聴感 A/B = **区別不能**（ユーザー確認 2026-09-01）
- case.no-ref（対照）: SNR 100.5 dB

## 裁定と反映（2026-09-01・ユーザー承認）

- ズレの改善は不要: 種は浮動小数の実務的な床（演算順の差）で、増幅率はモデル自身の性質。
  上流の演算順の逐語再現は WebGPU 向けグラフ設計と衝突し、効果もない。
- 検証は **2 段化**: 1 段目 = 従来の固定 1e-3（fast path・挙動不変)。超過時のみ増幅率を
  実測し `worst ≤ amp × 5e-6`（10 セル実測上限の約 2 倍）かつ絶対上限 5e-2（式の取り違え =
  O(1) 級を止める）で合否。増幅率は meta.json の `sensitivityAmp` に記録される。

## 方法

`pipeline_ref.emit` の monkeypatch プローブ（許容を inf 化 + `_euler` を摂動つき 2 回走行に
差し替え）。台本は揮発 scratchpad（probe_f16_divergence.py / ab_wav_sensitivity.py）— 恒久版は
`run_case(sensitivity_probe=True)` として本体に取り込み済み。
