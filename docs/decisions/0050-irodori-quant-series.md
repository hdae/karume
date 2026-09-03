# 0050: Irodori 量子化系列 — 波 1（f16）の形と門

- Status: accepted（波 1 = f16。i8 / w8a8 は本 ADR を拡張する後続の波で裁定）
- Date: 2026-08-12
- 関連: ADR [0006](0006-quantization.md)（fake-quant 方法論）/ [0018](0018-f16-weight-execution.md) /
  [0027](0027-sbv2-f16-series.md)（f16 系列の先行形）/ [0029](0029-sbv2-i8-series-and-quant-quality.md)
  （系列×格納 dtype の検出限界）/ [0048](0048-irodori-host-port.md) / [0049](0049-irodori-codec-integration.md)。
  判断材料の recon = [research/2026-08-12-irodori-quant-recon.md](../research/2026-08-12-irodori-quant-recon.md)

## Context

Irodori-TTS v4 の配布形は f32 単一系列（8 グラフ・3.44GB）だった。量子化の受け皿（ランタイムの
f16/i8 格納実行・dist の quant 表・models の session 配線）は SBV2 / Anima で完成済みで、
不足はエクスポータの dtype 軸と品質門の設計だけ。recon の実測で 8 グラフ全ての重みバイトの
99.8〜99.9% が重みスロットのみ消費（= 圧縮適格）と確認済み。ユーザー裁定（2026-08-12）は
3 波構成 — 波 1 f16 → 波 2 i8 + 品質台本 → 波 3 w8a8。

## Decision（波 1）

1. **f16 は別系列**（`outputs/series/irodori-v4-small-f16/` / `dacvae-32dim-f16/`）。
   fake-quant は各台本の**実効重み確定後・参照/golden 採取前**に当てる — export_irodori /
   irodori_pipeline は `load_*` 直後（remove 相当は変換時に焼き込み済み）、export_dacvae は
   `fold_weight_norm` → 門 3 決着 → `lift_snake_alphas` → 丸め → golden 採り直し → 切り詰め、
   の順（門 3 は丸め前の参照で決着させるので主張は弱まらない）。**Snake の alpha は丸めない**
   （lifted 定数 = 重みスロット外・golden も f32 alpha で計算 — 両者は整合）。
2. **配布形のレイアウトを `<role>/model.{f32,f16}.safetensors` へ変更**（Anima / SBV2 と同形・
   未リリースにつき破壊的変更で正）。quant 席は f32 / f16 の 2 席・混成なし（f16 に S を動かす
   軸が無いため一律で落とせる — 混成が要るのは i8 の波）。defaultQuant は f32 のまま。
3. **系列×格納 dtype は両側から挟んで一意にする**。既存の存在検査（`assert_storage`）だけでは
   **f32 席への f16 資産の挿し込みが素通りする**（圧縮系列のコンテナも bias / norm / グラフ
   定数を F32 で持つため「F32 を含む」が真になる — SBV2 では 2 席とも圧縮系列でこの穴が露出
   しなかった）。`assert_storage_absent` + 禁止表（f32 席は F16 禁止）を新設し、逆向きは既存の
   存在検査が落とす。グラフ宣言との突合（`assert_irodori_graphs`）も全 dtype 系列に掛ける。
4. **latent 門は系列パラメタ化**し、tolerance は系列ごとに素の実測から独立導出（ADR 0027 の
   型・流用禁止）。**未導出の系列は SKIP ではなく「実測を出して赤」**（資産があるのに何も
   検証していない緑を作らない）。S / forwards の完全一致は系列に依らず要求する。
   実測（2026-08-12・実 GPU）: f16 は S/forwards が f32 と完全一致・z maxAbs は
   full 2.1321e-4 / no-ref 1.8516e-4 → **F16_Z_ATOL = 1.5e-3**（最悪の 7.0 倍）。
5. **WAV sha256 門は f32 専用のまま**（門自身が tolerance 化・参照差し替えを MUST NOT と
   宣言している）。f16 の最終裁定は聴感（ユーザー）— A/B は同一テキスト・同一 seed の WAV 対で
   提出する。上流突合（`EULER_REFERENCE_ATOL` 1e-3）は**系列共通**のまま — 丸めはグラフ経路と
   上流経路の両辺に同じ 1 回だけ当たるので量子化誤差は相殺し、測るのは f32 と同じ「実装差
   だけ」（f16 実測 2.1e-4 / 7.8e-5 で成立を確認済み）。
6. **i8 / w8a8 は後続の波**。分岐点は S ドリフト（duration を i8 にすると発話長が動きうる —
   SBV2 w8 の w_ceil 198→196 と同じ軸）で、混成表（duration 据え置き）の裁定は
   `measure_quant_irodori`（波 2 で新設）の実測後に行う。

## Consequences

- 配布 3,438,182,144 → f16 系列 1,721,312,568 B（**50.1%** — 実測 2026-08-12）・
  DiT 常駐 1.46 → 0.73GB。
  実測サイズは dist 出力（各 role の `model.f16.safetensors`）が正本。
- `karume dist --pipeline irodori` は**両系列が揃わないと組めない**（f32 単独の組み立て不可）。
  再組み立て時は旧レイアウトの残骸があると宣言外ファイル検査で落ちる（先に消す）。
- 計算は f32 のまま（格納だけ f16 — ADR 0018）なので速度利得はロード/VRAM のみ。速度の軸は
  波 3（w8a8）で、着手前に DiT の linear/bmm 比の実測が要る（recon の risks）。
- **聴感裁定（2026-08-12・ユーザー）: f32/f16 の A/B 対（同一テキスト・同一 seed・通常合成 +
  声クローン）で「ほぼ同じ音声」— f16 系列を受理**。SBV2 f16（ADR 0029 の表）と整合する結果。

## 追記（2026-08-12）— 波 2 + 波 3 クローズ: w8 / w8a8 席と既定の確定

測定台本 `measure_quant_irodori.py`（9+1 構成 × 2 ケース・torch CPU sim）と実 GPU の実測で
決定 6 の分岐を解消し、量子化 3 波を閉じた。

### 実測（判断の根拠）

- **S ドリフト = 全構成・全ケースで不変**（`i8-duration-only` は z も波形も基準と完全ビット
  一致 — 銀行家丸めの境界を踏まなかった）。決定 6 が警戒した混成の必要が実測で消えた。
- 品質（sim・基準 f32 と同一時間グリッド）: f16 = SNR 37.0/52.0dB・LSD 0.38/0.11（透明）/
  w8（全 i8）= SNR 15.6/6.75dB・LSD 2.24/2.18 / **w8a8 = SNR 2.54/7.17dB・LSD 5.64/2.10**
  （full 側の数値劣化が大きい）。劣化の帰属は直交分解で **DiT が支配**（i8-dit-only ≈ i8-all）。
- 実 GPU: latent 門 w8 席の素実測 = full 1.8429e-3 / no-ref 1.6475e-4（S/forwards 完全一致）。
  w8a8 の w8 golden との z maxAbs = 2.97/1.43。速度 wall ×1.12（8.5 → 7.6s・同一条件 2 回）。
- **聴感裁定（ユーザー・DAC + ヘッドホンで f32/f16/w8/w8a8 を通し確認）**: 「それぞれ音質的な
  劣化という感じはしない」・w8a8 の差は「末尾の雰囲気が変わった？」程度 — **数値の帯より
  聴感が正**（Anima w8a8「別の絵だが受理」ADR 0025 と同じ帰結）。全 4 席を受理。

### 決定（追補）

7. **quant 席は 4 つ**: f32 / f16 / **w8 = 全 i8 一律**（混成なし — S 不変の実測とユーザー
   裁定）/ **w8a8 = w8 とバイト共有** + `session {"linearCompute":"i8a8"}`。席名 →（格納
   dtype・session）の対応は `IRODORI_QUANT_SEATS` の 1 箇所だけで綴る。
   **既定 = `w8a8`**（聴感裁定 — 配布 25.2%・DiT 常駐 0.37GB・wall ×1.12 が既定で効く）。
8. **f32 席の禁止表は圧縮 dtype 集合 {F16, I8}**（i8 資産も bias / scale を F32 で持つため、
   存在検査 + F16 禁止だけでは素通りする — 波 1 の穴の i8 版）。
9. **w8a8 の門は数値パリティ網にできない**（ADR 0025 決定 6）— `e2e_irodori_w8a8_test.ts` が
   ①S / forwards / latentDim の w8 golden との完全一致 ②判別帯 zBand [0.1, 6]（下限 = w8 席
   実測の 54 倍 — 沈黙 f32 フォールバックは差が縮む側に出る〈ADR 0028 決定 6〉）③キー census
   （i8a8 linear 317 / quantize_rows 317 / **素の linear 0** — `lastRunTiming` 観測・計測無効
   run は fail loudly）で縛る。latent 門の w8 席（W8_Z_ATOL 1e-2 = 実測の 5.4 倍）と併存。

## 追記（2026-08-23〜24）— 波 J-4: `w4` 席（DiT のみ i4 の混成・GPTQ 校正付き）

品質イテレーション 3 ラウンド（R1〜R3）と裁定の実測記録は
[research/2026-08-24-gptq-expansion-quality.md](../research/2026-08-24-gptq-expansion-quality.md)。

### 過去裁定の変更 2 件（2026-08-23 ユーザー承認）

- **決定 7 の「混成なし」を緩和**: 当時の根拠 S ドリフトは GPTQ で消滅
  （[research 2026-08-20 §6](../research/2026-08-20-gptq-awq-calibrated-rounding.md) の
  S 完全一致）。DiT のみ i4 の混成を席として認める。
- **「rtn 席を先行させない」（2026-08-20・kmeans 圧勝を受けた裁定）を撤回**: GPTQ 適用拡大
  （幅）を優先する再定義。kmeans 席（perf-ledger Q-2）は席の追加で後から共存できる。

### 決定（追補 2）

10. **quant 席は 5 つ**: 既存 4 席 + **`w4` = DiT のみ i4 g32 の混成**。DiT コンテナの内訳は
    **block 内の adaLN 以外 168 本 = GPTQ i4（gptq-rtn 格子・校正 12 件 × 40 step）/
    i8 149 本 = adaLN 144（`attention_adaln` / `mlp_adaln`）+ block 外 5（`in_proj` /
    `out_proj` / `cond_module.{0,2,4}`）/ bias・norm f32**。他 7 役は w8 の i8 系列と
    バイト共有・session = {}（`linearCompute` は宣言しない — w4a8 は irodori 未測定・anima は
    品質で不採用〈ADR 0076 決定 6〉）。**既定は `w8a8` 据え置き**（w4 は温間合成が最遅
    +38% — 存在理由はサイズと帯域）。
11. **i4 の 3 分割は聴感の帰属実測で確定**（research §1）: block 外 5 本の素 RTN i4 と校正
    コーパスの汎化不足がこもりの実因（→ i8 へ / 12 件へ増量・評価入力と部分一致まで分離）、
    adaLN は量子化感度が高く i4 のままだと韻律がずれる（→ i8 へ・+13.1 MiB）。adaLN 判定は
    `irodori.calib.is_adaln` の 1 実装を校正 include と格納分割で共有し、**adaLN 不在は
    fail loudly**（上流改名で裁定済み構成が黙って壊れるのを防ぐ）。
12. **golden は出荷バイトから焼く**: `pipeline_ref --dtype i4` は出荷コンテナ読み戻し
    （`restore_dit_from_i4_series` — provenance = gptq 門・形 / 本数 / 席の効き門つき）。
    latent 門の w4 系列は **W4_Z_ATOL 6e-3**（R2 実測 full 1.1522e-3 の 5.2 倍で導出・R3 実測
    4.7213e-4）。校正 provenance（`calib_provenance.json`）は dist の plan 時に無条件検査
    （method == "gptq" 必須 — 素 RTN 系列の混入を塞ぐ）。
13. **sim → 出荷の転移限界を品質裁定の前提にする**（research §2）: 計測リグの A/B sim が
    証明するのは同一リグ内の効果まで — 出荷リグでは GPTQ の丸め解が変わり発話実現が再抽選
    される（z relRMS がセル間距離級）。**繊細な性質（韻律の実現）は sim から出荷へ転移しない**
    前提で、最終裁定は必ず出荷バイトの聴感で行う。adaLN i8 の出荷リグでの寄与は未検証のまま
    クローズ（載せ外し A/B は実需待ち — backlog）。
14. **聴感裁定（2026-08-24 ユーザー）: R3 = こもり解消・配布可（opt-in）**。HF 公開済み
    （コミット `67e9584c`・断片化 29.4 MiB/レンジ健全）。**pin は据え置き** — `w4` を使うには
    `revision: "main"` の明示が要る（anima i4 と同じ形）。

## 追記（2026-08-25）— 0.5.0 で pin 解消・席名は `i8+dit4` へ

決定 14 の「pin は据え置き」は 0.5.0 リリースで解消した — `IRODORI_V4_SMALL_CURRENT` が
席名改名後の revision を指し、`revision: "main"` の明示は不要になった。席名は
ADR [0074](0074-quant-seat-naming.md) の移行表どおり `w4` → `i8+dit4`。
