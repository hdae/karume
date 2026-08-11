# Irodori-TTS v4 ソース精読 recon（karume 移植可否の確定）

> 時点スナップショット（2026-08-11）。GitHub `Aratako/Irodori-TTS`（HEAD `8224daf`）+
> `facebookresearch/dacvae` のソース精読と、実重み safetensors ヘッダ / codec pickle メタの
> range 取得による実測（重み本体は未取得）。先行 recon
> （[2026-08-11-model-expansion-recon.md](2026-08-11-model-expansion-recon.md) §Irodori —
> 確度 medium/low）の検証を含む。file:line は当該 HEAD 時点の値。

## 0. 結論

**技術的には移植可能。新規 IR op は `sin` 1 本だけ**（DACVAE の Snake 活性
`x + (α+1e-9)⁻¹·sin²(αx)` — x が実行時値で畳めず、既存プリミティブで合成不能・Core ATen 内
→ ADR 0043 の第 1 層）。残りの op 課題は全て第 0 層（patch / 分解 / ホスト昇格）で消える。
工数の重心は DiT ではなく **codec decoder のタイル化**と **tokenizer の 4 つ目の実装**。

**ユーザー裁定 2 件（2026-08-11・着手時の前提）**:

1. **CFG マスク = 実行時 bool マスク入力を許す（案 a）**。uncond を「state=0 + mask 全 0」で
   作る Irodori の CFG は、実行時 attention マスクを持たない現行設計（ADR 0016 のガード不活性
   証明がマスク実値の静的評価を要求）と正面衝突する。「state=0 + mask=1」代用は
   `exp(0)=1` の重みが残り数値非同値。ガード不活性の証明を実値評価以外（構成的保証）へ
   置き換える設計裁定で、**着手時に ADR 化**。EmbeddingGemma の「実行時 attention_mask
   非対応」（limitations）も同じ一手で解消される。
2. **SilentCipher 透かし = 公式準拠で基本入れる**（payload "IRDTS"）。無効化してよい種類の
   ものならフラグ制御（既定 on・明示 opt-out）。実装自体は独立した外部 NN の後処理段なので
   **公開前の波まで保留可** — パイプライン API に席だけ確保する。

## 1. 構成（実重みヘッダ実測・合計 766,052,385 params / 714 tensors）

| 群                         | params | 要点（実装は repo の `irodori_tts/model.py`）                                                                                                                                                                                   |
| -------------------------- | -----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Joint-Attention DiT × 12   | 358.4M | dim 1280 / heads 20（head_dim 64）/ SwiGLU（×2.875）/ **LowRankAdaLN**（rank 192・重み無し RMS・tanh ゲート）。K/V は `[self｜text｜speaker｜caption]` の cat・条件側は専用 wk/wv・RoPE は**先頭 10 head のみ**・sigmoid ゲート |
| ModernBERT-ja-310m（同梱） | 314.6M | 25 層 / hidden 768 / GeGLU（**erf 形 gelu**）/ LayerNorm（bias なし）/ 3 層毎 full + sliding 128（半幅 64）/ RoPE θ 160000/10000 の 2 系統。text/caption で backbone 共有・projector 別                                         |
| ReferenceLatentEncoder × 8 |  60.5M | latent 32×patch 4 = 128 → 768。q/k に rank-2 weight の RMSNorm・complex 形 RoPE・sigmoid ゲート付き attention                                                                                                                   |
| DurationPredictor          |  21.8M | SwiGLU ブロック ×3 + speaker/caption の AdaRN-Zero 変調。出力 = `log1p(Σ softplus·mask)`                                                                                                                                        |
| cond_module ほか           |   8.9M | timestep → AdaLN 条件（512 → 3×rank）等                                                                                                                                                                                         |

コーデックは別リポ・別重み: `Aratako/Semantic-DACVAE-Japanese-32dim`（MIT・`weights.pth`
429.6MB — **torch pickle なので safetensors 変換台本が要る**）。48kHz / hop 1920 = 25Hz
フレーム / VAE latent 32 次元。decoder 主経路は Snake1d + ConvTranspose1d + ResidualUnit
（dil 1/3/9）×4 ブロック（1536→96）+ Tanh 波形化。**ELU / LSTM は透かし枝専用で主経路に
出ない**（Irodori 側が `alpha=0` で無効化して `forward_no_conv` へ差し替え）。

## 2. 推論フロー（正本 = `inference_runtime.py:1036-1462`）

text 正規化（NFKC ほか）→ tokenize（BOS 前置・右詰め pad）→ 条件エンコード 1 回 →
duration 予測 → **`.item()` で S 確定（ホスト読み戻し — SBV2 と同型）** → context KV
事前射影 1 回（12 層 ×(k,v)×3 セグメント = 72 本）→ **Euler 40 step ループ（ホスト駆動・
CFG は前半 t∈[0.5,1.0] のみ・independent 3 本で forward バッチ 4 倍）** → unpatchify →
codec decode → 無音末尾検出 → SilentCipher 透かし。

可変軸は 4 本: Tt≤256 / Tc≤512 / Ts≤750 / S≤750。JointAttention の K 長は
`S + Tt + (Ts+1) + Tc` の合成。caption 未指定・参照なしでは**全 0 マスクが正規に出る**
（→ §0 裁定 1）。

## 3. export 分割案（6+2 グラフ・B=1）

G1 ModernBERT backbone（T sym）/ G1a・G1b projector / G2 speaker encoder / G3 duration
（系列入力なし — speaker_vec・caption_vec はホストで切り出せる）/ **G4 context-KV**（40 step
不変 → 1 回計算して 160 forward で再利用 = **prepared 機構〈ADR 0042〉の 2 例目**）/
G5 DiT 1 step / G6 codec decode（+ G7 encode は cloning 用）。ホスト残置は Euler 更新
（`[1,S,32]` ≤ 24k 要素）と CFG 合成のみ。

第 0 層で消す 6 件: ① complex RoPE → 実数対形 patch + 表のホスト供給（patch_anima と同型）
② timestep sin/cos → `t_embed` 入力昇格 ③ 重み無し RMS → ones weight 挿入 or fold 拡張
④ rank-2 weight RMSNorm → rms_norm + mul 分割 ⑤ `F.softplus` → gt_scalar/where/log1p/exp
分解（torch 突合ゲート必須）⑥ weight_norm → export 前に焼き込み。SDPA + bool mask は
既定 preserve の分解が正解（融合 attention は mask を取れない — EmbeddingGemma と同じ判断）。

## 4. tokenizer / 前処理

`sbintuitions/modernbert-ja-310m` = **Unigram（vocab 102400）+ byte_fallback**・normalizer
null・Metaspace（`prepend_scheme="never"`・`split=false`）。T5 実装から Viterbi 本体は流用
できるが、prepend_scheme / split / byte_fallback の 3 点差で **models 4 つ目の新規ファミリ**。
前処理は NFKC 正規化 + 記号置換 + BOS(id=1) 手前置・右詰め pad(id=3)（post_processor の
`</s>` は付かない）。絵文字スタイル制御 56 種は byte_fallback 経由になり得る。

## 5. リスクと未知（U 番号は未実測）

- **U1**: `@torch.jit.script` の snake が torch.export を通るか（通らなければ同値関数へ差し替え）
- **U2**: ModernBERT の rope_parameters バッファが FOLDABLE の葉へ降りるか
- **U3**: 記号次元 4 本同時（Tt/Tc/Ts/S）を karume が扱えるか — 未確認。_推測_: 各セグメント
  Tmax 固定 + S のみ sym が最小手
- **U4**: DACVAE decoder の中間容量 _推定_ S=750（30s）で **~553MB** → タイル化が要る公算
- **U5**: 蒸留版 codec のテンソル鍵全件照合（重み未取得のため未実施）
- **U7**: 40 step × CFG 4 倍の実測レイテンシ（DiT 1 step = 12 層 × K≈1000 × dim 1280 を 160 回）
