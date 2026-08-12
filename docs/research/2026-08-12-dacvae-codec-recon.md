# DACVAE codec（G6/G7）recon — 第 4 波の設計材料

> 時点スナップショット（2026-08-12・実重み `inputs/irodori/dacvae-32dim` + 実装 clone
> `inputs/irodori/dacvae-src`〈facebookresearch/dacvae @ 414c2078〉に対する静的読解 + 部分
> export・受容野・メモリの実測）。裁定の帰結は ADR
> [0049](../decisions/0049-irodori-codec-integration.md) が正本。file:line は読解時点のもの。

## 1. decoder（G6）の実体

- 経路: `quantizer.out_proj`（conv1d 32→1024 k=1）→ conv1d(1024→1536,k7) → Snake →
  DecoderBlock ×4（convT k=2s / s = 12,10,8,2 + ResidualUnit d=1/3/9 ×3）→ Snake(96) →
  conv1d(96→1,k7) → Tanh。**波形ヘッドは wm 枝の前段**（`wm_model.encoder_block.pre`）が
  兼ねており、バイパスは README ②形（`forward_no_conv` 差し替え）が必須 — `alpha=0` だけ
  では 96ch のまま返る（実測）。`DecoderBlock.forward` は `_chunk_size` の倍数のかたまり
  だけを通し、ELU/causal 系の残り（up/down サンプラ）は `watermark()` からしか呼ばれない死枝。
- torch.export の aten op 実測（S=8）: conv1d 27 / conv_transpose1d 4 / sin 29 /
  reciprocal 29 / pow² 29 / tanh 1 / add / mul / reshape。`remove_weight_norm`（30 本）で
  `aten._weight_norm` は消え出力はビット一致。**convT 4 本は karume の受理形
  `2·padding == K − stride` を全て満たす**（k=2s・pad=s/2）— ランタイム無変更で載る。
- Snake: `x + (α+1e-9)⁻¹·sin²(αx)`・α は per-channel（58 本・min −0.347 / max 4.99・
  min|α+1e-9| = 2.225e-5 → 畳み込み定数 44944）。`(α+1e-9)⁻¹` は **lifted 定数化**で
  第 0 層畳み込みに載る（Parameter のままだと `convert._classify_foldable` の対象外で
  reciprocal が実行時 op に残る）。
- **sin の引数は π 超えが常態**: 実 z で |α·x| は decoder 最大 15.9（≈5.06π・波形ヘッドの
  Snake(96)）/ encoder 最大 71.3（≈22.7π・先頭 Snake(64)）。GPU 実測誤差は 5e-6 級に留まり
  引数簡約が f32 精度を保つ（初観測 — `e2e_dacvae_test.ts` の docstring が正本）。

## 2. メモリと タイル化

- S=750: block3/4 の中間 [1,192,720000] / [1,96,1440000] = **527MiB**（残差で 2 本同時 ≈
  1.03GiB）。単一バッファは 737,280·S バイト → **既定 128MiB 上限の機は S ≥ 183 で確保失敗**
  （本機 2GiB−4 は単発 10 秒で完走）。encoder は 120 秒（T=3000）で **1.47GB×2**。
- 受容野（デルタ摂動の実測）: decoder 片側 13,793 サンプル = 7.19 フレーム → **halo 8**。
  encoder 21,826 サンプル → halo 6。主経路は因果層ゼロ・平行移動同変なので「halo を捨てた
  タイル内部は全長とビット一致」— f64 で HALO=8 → 1.56e-17（打ち切りゼロ）を実測。karume の
  conv カーネル（出力要素ごと固定縮約順の gather）では **Uint32 完全一致**が門として成立
  （`e2e_irodori_codec_test.ts` 実測）。
- 上流にストリーミング decode は無い（`CodecMixin.compress` のチャンク機構は padding=False
  前提の DAC CLI 用別方式で Irodori は使わない）。

## 3. encoder（G7）と in_proj

- conv1d 30 + Snake 29 + 残差 12 + `in_proj`（1024→64 k=1）— **decoder 語彙の部分集合**
  （convT なし）。決定的 encode は `in_proj(...).chunk(2,dim=1)[0]`（mean 側 32ch）だけを
  使う → **前半 32 行 + bias 前半 32 に切り詰めた conv1d と厳密同値**（export の常設門）。
  rng 無し = 純関数。
- `_pad`: 右側のみ・端サンプルを含まない reflect・hop 1920 の倍数へ（ゼロ pad ではない）。
- IR の束縛規則（シンボルは入力 shape に**素の形**で現れる）により入力は `[1,1,1920T]` に
  できず **`[1,T,1920]`**（要素順を変えない読み替え・ホストは連続バッファをそのまま渡せる）。

## 4. 参照音声のホスト前処理（上流 codec.py:132-254）

mono 平均 → resample（**48kHz 入力ならスキップ** — sinc_interp_hann 多相 FIR は未移植・
軸④裁定で fail loudly）→ **ITU-R BS.1770-4 LUFS**（K-weighting = high_shelf(4dB, 1/√2,
1500Hz) → high_pass(0.5, 38Hz) の RBJ biquad 2 段・厳密 IIR・0.4s/75% ブロック・絶対 −70 +
相対 −10 ゲート・0.5 秒未満はゼロ pad 測定・−70 clamp）→ gain = 10^((−16−refDb)/20) →
ensure_max（peak>1 のみ 1/peak）→ reflect pad。

- **ブロックの数え方は `julius.core.unfold`**（audiotools の `Meter._unfold` — フィルタ後に
  末尾ゼロ pad・F = 1+ceil((T−19200)/4800)）。`torch.nn.Unfold`（末尾切り捨て）と読み違えると
  端数長の入力だけ LUFS が 0.074 LU ずれる（TS 実装時に一度踏んで実測訂正 — その段差自体を
  恒真化遮断の門にした）。
- 正規化は「LUFS 利得 × peak 利得」の 2 本に**厳密に分解できる**（`dacvae_host.py` が上流
  出力とのビット一致を毎 emit 実測）。TS は f64 で LUFS を回し、f32 `lfilter` との差
  refDb 2.2e-5 LU 級（利得相対 2.5e-6・波形 6e-7 = −106dBFS）を parity 門の閾値に載せる。
- ref-hot（振幅 3 倍）は **LUFS 経路では peak 制限を踏まない**（−16 正規化で 0.81 へ下がる）。
  LUFS 経路の peak 制限を踏む golden は ref-short のみ。
- WAV の int16 → f32 は **/32768**（soundfile 実測・合成 wav の両端で確定）。torchaudio 側は
  本環境で実測不能（torchcodec/FFmpeg 欠如）— 歴史的には同じ /32768。

## 5. trim_tail と切り出し

`find_flattening_point` は **z（latent）上**で判定（窓 20 行のゼロ pad・窓全体の母標準偏差
< 0.05 かつ |mean| < 0.1 の最初の index）。`flattening_samples = point × 1920`・
`flattening > 0` のときだけ `min(targetSamples, flattening)` で**波形を**切る。latent を
切ってから decode すると境界 padding が変わり全長 decode とビット一致しない（実測）—
decode は常に全長。

## 6. open（第 4 波クローズ時点）

- encoder のタイル化（halo 6 で成立するはずだが平行移動同変性の実測は未実施）— 長い参照 ×
  既定上限機で必要になったら decoder と同じ形を足す。
- `WAVE_FORMAT_EXTENSIBLE`（0xFFFE）の受理（SubFormat GUID の読み替えで PCM16/f32 に
  落とせる — 実サンプルは素の PCM16 で不要だった）。
- resample の移植（軸④で見送り — 仕様は recon 済み: torchaudio v2.8 functional.py:1411-1532）。
- LUFS 経路の peak 制限を ref-short 以外でも押さえる golden の追加（dacvae_host.py に
  「短くない + 小音量 × 大利得」ケースを 1 本足す案）。
