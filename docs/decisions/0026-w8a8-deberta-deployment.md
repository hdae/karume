# 0026: w8a8 の DeBERTa 展開（他モデル検証）

- Status: accepted
- Date: 2026-08-03
- 関連: ADR 0019（i8 格納）/ ADR 0025（w8a8 実行 — 決定⑤が本 ADR の親）/
  [dp4a-w8a8-design](../research/2026-08-03-dp4a-w8a8-design.md) §7

## Context

ADR 0025 決定⑤: w8a8 の「他モデル検証」の受け皿は DeBERTa full-24layer（linear が FLOP の
86.8% — SBV2 は conv1d 支配で構造的に不成立）。ランタイムには opt-in の
`SessionOptions.linearCompute: "i8a8"` が既にあり、必要なのは i8 資産系列と検証網・
品質/速度実測の展開だけ（ランタイム変更 0 行で成立するかが一般性検証の中身）。

## Decision

1. **資産系列**: `export_deberta.py --dtype {f32,i8}`（f16 は #30 SBV2 系列と一体で決める
   ため足さない）。i8 は別系列 `models/deberta-i8/{dev-2layer,full-24layer}/`
   （24 層 334,545,336 B = f32 比 25.4%）。fake-quant は export する wrapper そのものへ・
   golden 採取より前（quantize.py の FQN 規律）。
2. **w8（f32 計算）の網**: `packages/runtime/tests/e2e_deberta_test.ts` を系列パラメタ化。i8 系列 tolerance は
   実測導出 atol 7e-4（素の最悪 1.23e-4 の 5.7 倍）。f32 系列の 8.32e-5 と同桁に収まること
   自体が「golden が fake-quant 後の重みで採れている」ことの裏取り（掛け忘れなら量子化誤差
   そのもの = 1e-1 級が 3 桁上に出る）。
3. **w8a8 の網**: `packages/runtime/tests/e2e_deberta_w8a8_test.ts` 新設 — 鏡像 golden
   （`--act-quant` → `io-i8a8.<case>`・`act_quant.py` が数値正本）との突合 + 診断キー検査。
   数値の検出力は **output.0/1 の厳密 tolerance（atol 5e-5・f32 経路との余白 1,150×）に
   集中**させる（下記・検出限界）。
4. **品質ゲート**: 層別 SNR（数値・下記）+ SBV2 デモの聴感（最終・ユーザー裁定）。
   デモに `--bert {f32,i8}` / `--bert-compute {f32,i8a8}` ノブ（anima の
   `--dit` / `--linear-compute` と同型・i8a8 は `--bert i8` とだけ組める）。

## 実測（2026-08-03・RTX 3080 Ti・排他）

### 速度（T=512 合成トークン・full-24layer・warmup 1 + 5 run 平均）

| 構成                    | ロード  | GPU 合計   | linear カーネル     | 壁時計/run |
| ----------------------- | ------- | ---------- | ------------------- | ---------- |
| f32 資産                | 197.7ms | 125.7ms    | 93.7ms（74.6%）     | 291.7ms    |
| w8（i8 資産・f32 計算） | 62.3ms  | 126.9ms    | 93.0ms（73.3%）     | 289.9ms    |
| **w8a8**                | 62.3ms  | **72.4ms** | **33.2ms（45.8%）** | 251.6ms    |

- **linear 単体 2.80×・GPU 合計 1.74×**（Anima DiT の 3.25× / 2.25× より小さいのは m=T=512
  固定で GEMM が浅いため）。w8 は計算を速くしない（Anima と同じ — 効くのはロード 3.2× と
  資産 1/4）。
- 壁時計が 1.16× に留まるのは、このグラフが**全 25 層の hidden_states を出力に持ち**
  読み戻し ~52MB が支配するから（SBV2 用途の形。出力を絞る用途なら GPU 比がそのまま出る）。
- w8a8 時の次の支配項は bmm 18.6%（disentangled attention — 融合 attention は mask 契約が
  要るため対象外・ADR 0023 スコープ判断のまま）。
- SBV2 デモ実測（P=59・T=17）: bert 段 279ms → 228ms — 小 T ではロード/固定費支配で
  効果は限定的。w8a8 が効くのは長系列・スループット用途。

### 品質（GPU w8a8 vs w8 golden・full-24layer）

- SNR は約 1dB/層で単調低下（output.1 39.2dB → output.24 19.4dB・段差なし — 一様量子化
  雑音の層蓄積。Anima の層別実測と同じ性質）。
- **hidden[-3]（SBV2 が使う層）= 20.7〜23.3dB**（4 ケース・maxAbs 6.3e-1〜9.6e-1・
  |ref| 上端 16〜29）。参考: Anima の w8 PSNR 23.92dB / w8a8 13.28dB（後者をユーザーが
  目視受理）の中間より上。
- 聴感: WAV 3 構成（f32 / w8 / w8a8・同 seed）をユーザー裁定へ提出（2026-08-03）→
  **ユーザー受理（2026-08-04・「劣化は感じられない」）**。

## 検出限界（このタスクで確定した知見）

- **活性量子化の不連続性は encoder の 1 forward でも数層で飽和する**: GPU i8a8 vs torch
  鏡像は末端層で maxAbs 1.46 — f32 経路 vs 鏡像の 1.33 と同水準（= 判別不能）。ADR 0025
  決定 6 の「同じ分布の別標本」が拡散 10 step どころか 24 層 1 パスで再現した。判別帯は
  **1 層目の出口（output.1）にしか無い**（i8a8 5.72e-6 vs f32 経路 5.76e-2 = 10,058 倍）。
  「実測の 5〜10 倍」の素直な tolerance 導出は f32 経路まで飲み込んで恒真化する —
  数値の検出力は output.0/1 の厳密 tolerance と診断キー検査（192 本）に集中させた
  （導出表はテスト冒頭 docstring が正本）。
- **診断キーの期待本数はモジュール数と一致しない**: DeBERTa-v2 は `share_att_key` で
  query/key_proj を相対位置射影へ再利用するため、1 層あたりモジュール 6 本・グラフノード
  8 本（2 層 16 / 24 層 192）。torch のフックは呼び出しごとに掛かるので鏡像との数値対応は
  1:1 に保たれる。
- `rel_embeddings.weight` は fake-quant されるが linear の**入力**スロット消費 = 適格外で
  f32 格納（量子化済みの値・余剰 scale は emit が無視）。ConvLayer の conv1d weight は
  i8 適格（w8 実行・w8a8 対象外）。

  > 訂正（2026-08-11・実資産の IR を直接読んだ実測）: 消費 op は linear ではなく **`layer_norm`
  > の data スロット 1 箇所**（`get_rel_embedding()` が相対位置テーブルに LayerNorm を掛ける経路 —
  > `norm_rel_ebd="layer_norm"`）。「重みスロット以外の消費 = 適格外」という結論そのものは変わらない。
  > per-row i8 なら 512 行すべてビット一致で往復する（実測・最大差 0.0）が、適格判定を緩める
  > blast radius に見合わないため見送り —
  > [research/2026-08-11-deberta-size-recon.md](../research/2026-08-11-deberta-size-recon.md) §3。

## Consequences

- **一般性の確認は成立**: ランタイム 0 行変更・配線のみで第 2 のモデルに w8a8 が通り、
  linear 2.80× / GPU 1.74× が出た。opt-in（既定 f32）の設計は変更しない。
- SBV2 の実用（小 T）ではロード 3.2× と資産 1/4 が主な利得で、計算時間の利得は小さい。
- 品質の最終裁定（聴感）= **ユーザー受理（2026-08-04・劣化なし）** — w8 / w8a8 とも
  聴感ゲート通過。数値上は hidden[-3] SNR 20.7〜23.3dB（Anima の受理水準より上）。
- 残タスクへの申し送り: #30（SBV2 f16 系列）で `--dtype f16` を `WEIGHT_DTYPES` へ足す
  ときは `DEFAULT_OUT_ROOTS` に 1 行 — 形は準備済み。
