# 0030: 融合 attention の a8 化（QK/PV の活性 i8 + dp4a 整数内積）

- Status: accepted（目視ゲートも**ユーザー受理** 2026-08-04「変化は誤差レベル」=
  品質込みで確定）
- Date: 2026-08-04
- 検出器の現況（2026-08-15 追記）: 正本に指定した
  `packages/runtime/tests/e2e_anima_attn_i8a8_test.ts` は models 移行（旧構成の廃止）で
  **削除済み**。現行の検出器は `packages/models/tests/e2e_anima_test.ts` の PNG sha256
  完全一致門 + `packages/runtime/tests/gpu_attention_i8a8_test.ts` /
  `gpu_attention_pv_i8a8_test.ts`。**旧 E2E の tolerance / 床の記録は歴史値**として読むこと。
- 関連: ADR 0023（融合 attention — 3 dispatch 構造の土台）/ 0025（linear w8a8 — 量子化部品と
  atol=0 契約の前例）/ 0028（計算精度ノブ `ComputePrecision` の型）/ 設計 recon =
  [2026-08-04-attention-a8-design.md](../research/2026-08-04-attention-a8-design.md)（qk/stats/pv
  内訳の初確定・P̃ scale 縮退の発見・案 A/B と全裁定の根拠）

## Context

1024px turbo w8a8 の GPU 支配項は attention（46%・qk 44.2 / stats 7.3 / pv 48.5%）で、
attention は linear の約 1/3.4 のスループットしか出ていなかった。linear で実証済みの dp4a
経路（ADR 0025）を QK/PV へ広げる。ユーザー承認 2026-08-04（案 B 段階導入・Q0 品質シム先行・
PV NG 時は案 A 縮退）。

## Decision

1. **opt-in `SessionOptions.attentionCompute: "i8a8"`**（既定 `"f32"` 不変・`"f16"` と同一
   enum の排他値 — q/k/v は全て活性で格納軸を持たないため直積は存在しない。`shader-f16` を
   要求しない・資産の格納 dtype と直交）。デモは `--attention-compute i8a8`（出力名 `-attni8`）。
2. **①QK**: `quantize_rows` を q / k に無改変で 2 回（縮約軸 D が最内連続 = linear と添字が
   同型）→ `attention_qk:v2:i8a8:reg64x64{v4}:dp4a{Emu}`。半スケール（ADR 0023 の √scale）は
   **量子化前でなく dequant 側**へ 2 回（`(qs·scale)·(ks·scale)` を先に畳む MUST）。
   **GPU vs TS 参照 atol=0 が成立**（6 形状・NaN 行/列伝播・ゼロ行厳密 0・dp4a/エミュ完全一致）。
3. **③PV**: **P は非実体化のまま** A タイル充填で `qP = round(127·exp(S−m))` を作る —
   **P̃ の per-token scale は 1/127 に構造縮退する**（行内 max = exp(0) = 1。amax を取っても
   同じ値になる = 適応の余地ゼロ・除算ゼロ・clamp 不要）。V は既存 `strided` で Vᵀ 化 →
   `quantize_rows`（行 = (b,h,d)）で **per-column scale と dp4a の N 連続パックを同時に取得**
   （新カーネル 0 本。per-token は縮約軸上で scale が変わり数学的に不成立）。dequant は
   `f32(acc)·((inv·(1/127))·vs[d])` の先畳み。数値契約は **2 段分割**: 整数の手前の純関数と
   atol=0（V=単位行列を qP 読み出し器に使う）+ qP 生成は「±1 段のみ + 不一致率 < 2%」の門。
4. **②行統計は f32 のまま 1 バイトも変えない**。分母を量子化後総和にする案（設計 §4.3）は
   **Q0 実測で反証**（cross-attn O relRMS 17 倍・PSNR 19.4→16.0 — 鋭い行では「丸めで消えた
   質量ぶん縮む」f32 分母の方が正しい近似）→ 不採用確定。
5. **適格判定は段ごとに独立**（① は D%4==0・③ は N%4==0 — パック方向が段で違う）。満たさない
   段だけ f32 へ**沈黙縮退**（混成あり・linear の k%4 と同じ流儀）。**検出器はキー本数検査
   ただ 1 本**（E2E が厳密に固定 — 故障注入で「数値の帯も床も緑のまま」を実証済み）。
   N > 2^17 は fail loudly（`LINEAR_I8A8_MAX_K` 輸入）。
6. **品質ゲートは torch シム先行**（`measure_quant_anima.py` 構成 (d)〜(h)・粒度はランタイム
   実装粒度と厳密一致 MUST）。512px: P̃ 行 peak/rms 中央値 12.35 ≤ 16・latent 倍率 ≤ 1.185×
   （vs 受理済み w8a8）で通過。**1024px は分布が右方シフト**（attn1 median 29.4 = 512px の
   1.84 倍・構造上限 √N に張り付く行が実在）したため **latent 門を 1024px 直接実測で再判定**
   → qk 0.960× / pv 1.084× / qkpv 1.066×（≤ 1.5×）で通過。正本 =
   `models/anima-demo/q0-attn-a8{,-1024}/`。最終ゲートはユーザー目視（ADR 0025 と同型）。
7. スコープ = **Anima DiT のみ**。VAE decoder は対象外（attention 5.65%・画質直撃）。
   SBV2 / DeBERTa は融合 attention op 不使用で**受け皿が構造的に存在しない**（ADR 0025
   決定⑤の「他モデル検証」に相当するものは無い — 検証は DiT 1 本に閉じる）。
   メモリ利得はゼロ（S は f32 のまま — メモリの手は ADR 0028 の f16 受け渡し）。

## 実測（2026-08-04・1024px turbo `--dit i8 --linear-compute i8a8`・seed 42・A/B/A/B 排他）

熱でクロックが振れる（linear 実測 958〜1,373ms/step）ため、**同一 run 内の linear を時計
プロキシにした比**で読む。妥当性: 対ごとの正規化一致 < 0.2%（w8a8 2.3824/2.3819・attn
1.6523/1.6498）・不変段 stats の比 ±8%。

| （·linear 比） | qk        | stats | pv        | 3 段計 | 量子化+Vᵀ 込み | DiT GPU    |
| -------------- | --------- | ----- | --------- | ------ | -------------- | ---------- |
| w8a8（f32）    | 0.526     | 0.083 | 0.584     | 1.193  | 1.193          | 2.382      |
| + attention a8 | 0.172     | 0.082 | 0.170     | 0.424  | 0.466          | 1.651      |
| 倍率           | **3.05×** | 1.0   | **3.43×** | 2.81×  | **2.56×**      | **−30.7%** |

- recon の見積り（中庸 2.0× → −20.4% / 上界 3.44× → −29.3%）に対し**実測は上界側**。
  linear の 3.44× に「dequant 除去分」が乗っていた分の割引を、量子化オーバーヘッドの小ささ
  （+0.042·lin ≈ 40ms/step、見積り 22ms の 2 倍だが桁は同じ）が相殺した形。
- 512px は attention 比重が小さく DiT GPU −4% 級。壁時計は非 GPU 固定費（~350ms/step）と
  熱揺れが支配的で、本セッションでは 34.9 → 34.2s（dispatch 3,087 → 3,311/step）。
  NOTE: 非 GPU 固定費は測定条件で 2 倍動く — 単発冷機の 1024px では 695〜704ms/step を実測
  （ADR 0031 Consequences）。「解像度非依存・dispatch 数支配」の性質は両測定で一致。
  **但し書き（2026-08-04・ADR 0032）**: この「非 GPU 固定費」の正体は帰属 recon で分解
  された — 約半分は GPU 時間診断の装置代（当時は自動有効・step とともに単価成長）、残りが
  ホストエンコード代で、submit ごとの `onSubmittedWorkDone` の直列化により壁時計へ全額
  乗っていた。ADR 0032 で両方解消済み — 本節の数値は当時の既定（計測 on・直列化あり）の
  値として読むこと。
- E2E（512px・素の実測から帯を導出）: attention a8 単独は f32 対照と **1,969 倍**離れる
  （床 3e-2 が opt-in 沈黙失効の数値検出器）。実運用形（w8a8 併用）は対照と **1.35 倍**しか
  離れない — 床は構造的に置けず、opt-in 検出はキー本数検査 1 本
  （`packages/runtime/tests/e2e_anima_attn_i8a8_test.ts` 冒頭が正本）。
- 既定経路: **PNG sha256 門 2 系列とも完全一致**・WGSL スナップショット・既存 tolerance 不変。
- verify 654/0/4・pytest 1,873（メイン自己実測）。

## 検出限界・知見（本タスクの新記録）

1. **P̃ の scale は構造縮退する** — 「P∈[0,1] だから量子化しやすい」は半分だけ正しい。値域は
   狭いが行内ダイナミックレンジが広く、品質は行の peak/rms が決める（`relRMS ≈ (peak/rms)/440`
   が実測と 84% 一致）。分布は N とともに右方シフトする（512px の門通過を 1024px へ外挿しない）。
2. **qP の GPU/TS 不一致は本機で 0/62,088** — 「exp の実装差で必ず割れる」想定は反証され、
   ±1 段 assert は実質恒真。**実働の検出器は不一致率の上限**（126 格子注入を 3.15% で赤化）。
3. **沈黙縮退（D%4 / N%4）はキー本数検査だけが検出器**（注入で数値の帯・床とも緑のままを実証）。
4. **P̃ タイルの片側だけの範囲外 0 埋め破りは検出不能**（相手側の 0 が相殺 — ADR 0022
   検出限界①と同型。両側同時は赤）。
5. **head 間で「大きさだけ」違うテストデータは payload base の取り違えを検出できない**
   （量子化後整数が一致するため。head 別**パターン**が MUST — テスト doc に固定）。
6. stats 分母の量子化後総和は理屈上きれいだが**実測で悪化**（決定 4）— 「分子だけ縮む系統
   バイアス」は実は消えた質量の真の寄与（≈0）に近い正しい近似だった。

## Consequences

- 1024px w8a8 運用の GPU 支配項だった attention が解消（−30.7%）。次の支配項は再び
  linear（1.0·lin = 61%）で、以降の手は帯域・非 GPU 固定費側（案 γ = 中間バッファ f16 格納 /
  Vᵀ+列量子化の 1 カーネル融合で dispatch 56 本回収など — recon §11）。
- attention_stats は attention 内 22.6% へ相対上昇（a8 で qk/pv が縮んだ結果）。f16 化で
  1.45〜1.58× の実測（ADR 0028）があるが S 格納形の互換が無く i8a8 と組めない — 記録のみ。
- 画像は f32 / w8a8 と別物（軌道分岐）だが、**目視ゲートはユーザー受理**（2026-08-04
  「変化は誤差レベル」— 512px / 1024px の before/after 4 枚で裁定）。w8a8 の「別の絵」
  （ADR 0025）より知覚差が小さい、という Q0 の相対劣化測定（latent 倍率 ≤1.1×）と整合。
