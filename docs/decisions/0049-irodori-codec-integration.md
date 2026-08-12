# 0049: Irodori codec 統合 — タイル分割・wm バイパス・LUFS ホスト・配布同梱

- Status: accepted（ユーザー裁定 2026-08-12 — 第 4 波計画の軸 4 つを一括承認 +
  試聴確認「ちゃんと聞き取れる」）
- Date: 2026-08-12
- 関連: ADR [0048](0048-irodori-host-port.md)（第 3 波ホスト契約）/
  [0047](0047-irodori-dit-execution.md)（DiT 実行形）/ [0041](0041-manifest-v2.md)（manifest）/
  recon = [research/2026-08-12-dacvae-codec-recon.md](../research/2026-08-12-dacvae-codec-recon.md)

## Context

第 4 波 = DACVAE codec（G6 decoder / G7 encoder）でテキスト → 波形と参照音声 → 話者条件を
閉じる。制約が 3 つ: ①decoder の中間活性は S=750（30 秒）で 527MiB — WebGPU 既定の
`maxStorageBufferBindingSize`（128MiB）の機では **S ≥ 183 で確保に失敗**する ②上流の透かし枝
（wm_model・LSTM 込み 9.3M param）は IR に載らない ③参照音声の前処理は
ITU-R BS.1770-4 の LUFS 正規化（audiotools）で、グラフに載らないホスト計算。

## Decision

### 1. decoder のタイル分割はホスト側・halo 8 フレーム捨てで**ビット一致**

decoder 主経路は因果層ゼロ・全層が対称 pad か厳密 `L·stride` の convT = **平行移動同変**。
片側受容野の実測は 13,793 サンプル（7.19 latent フレーム）で、halo 8 フレームを両側に付けて
捨てれば、採用区間は全長 decode と**縮約順まで同一**（karume の conv 系カーネルは出力要素
ごとの固定順 gather）。実 GPU の門（`e2e_irodori_codec_test.ts`）が実 z を 4 枚に割って
**Uint32 完全一致**を毎回実測する。したがってタイル長（既定 182 = 既定上限の機でも通る
大きさ）は**結果を変えない性能ノブ**（`codecTileFrames`）で、halo だけが pipelineConfig の
モデル定数（`codecHaloFrames` — 受容野由来）。先頭・末尾タイルの端は真の境界なので halo 不要
（グラフのゼロ padding が全長実行と同じ役をする）。

- 却下: グラフ内で slice/cat 表現 — 巨大バッファをグラフが触り続け、確保天井の問題が解けない。
- **encoder は当面タイルしない**（非対称）。参照音声は数秒〜十数秒が普通で単発が通る。120 秒
  （T=3000）では中間 1.47GB×2 に達し既定上限の機で落ちる — limitations に起票。受容野実測
  （21,826 サンプル・halo 6 フレーム）は recon にあり、要るときに decoder と同じ形を足す。

### 2. 透かし枝のバイパスは上流 README の②形（波形ヘッド 4 本だけ残す）

`decoder.alpha = 0.0` **と** `decoder.watermark = wm_model.encoder_block.forward_no_conv` の
両方を当てる。①だけでは 96ch のテンソルがそのまま返り波形にならない（波形ヘッド
Snake(96)→conv1d(96→1,k7)→Tanh が wm 枝の前段を兼ねている）。主経路の抽出とパッチ済み
Decoder の `torch.equal` を export の常設門にし、「wm_model 29 本中 25 本は decode に寄与
しない」を毎 emit 実測する。**生成音声に透かしは入らない**（SilentCipher は 2026-08-11 裁定
どおり公開前の波まで保留 — API の席も設けない〈死んだオプションを作らない・未リリースなので
後から足せる〉）。

### 3. codec は配布形へ同梱（karume-irodori-v4-small は 8 グラフ構成）

上流では別リポの DACVAE を、テキスト → 音声が 1 リポで完走するよう同梱する（+365MB f32）。

- 却下: karume-dacvae 別リポ（他モデルと共有可能）— hub の複数リポ参照（ADR 0038 §7）が
  **未実装**で、設計と実装が丸ごと要る。DACVAE を使う 2 本目のモデルが現れた時点で §7 を
  設計する — 未リリースなので分離は破壊的変更で可能。
- `sampleRate` / `hopLength` は codec の `metadata.json` から導出し、`sampleRate ==
  frameRate × hopLength` の整合を **TS parse 時と dist 組み立て時の両方**で検査（別 hop の
  DACVAE を混ぜた取り違えは shape が合ったまま別の声になる）。

### 4. LUFS 正規化はホスト移植・末尾トリムは z 上で判定

- 参照音声の前処理（`decodeWav` → 120 秒切り詰め → **LUFS −16 正規化** → peak 制限 →
  reflect pad）は全てホスト純関数。LUFS は BS.1770-4 の完全移植（K-weighting 係数は RBJ 式
  から生成・ブロックは `julius.core.unfold` の綴り = 末尾ゼロ pad）で、**f64 で回す** —
  上流の f32 `lfilter` を fround で写してもビット一致は出ないため、差（refDb 2.2e-5 LU 級 =
  WAV に書くと消える大きさ）を parity 門の閾値に載せる方を採った。
- 末尾トリム（`find_flattening_point`）は **z（latent）上で位置を決め、波形をサンプル単位で
  切る**。latent を切ってから decode すると境界 padding が変わり全長 decode とビット一致
  しない（recon 実測）— decode は常に全長。
- WAV の読みは **/32768**・書きは **×32767** の非対称を**意図として固定**（それぞれ上流の
  読み手と聴き比べ相手の torch 台本に合わせた結果 — 揃えると参照音声の LUFS が上流とずれる）。

### 5. E2E は latent 門と WAV sha256 門の**併存**

第 3 波の latent 門（z の tolerance 突合）は**残す** — ホスト経路（Euler+CFG+S 決定）を
codec と無関係に切り分ける唯一の層で、WAV 門だけでは障害時に「どちらの波が壊れたか」が
分からない。WAV sha256 門は移植の最終門（1 ビットの回帰検出・参照環境専用 — limitations の
既存規約）。ACTIVE_DESIGN が書いていた「WAV 門へ**置換**」はこの ADR で「併存」へ改訂。

### 6. 参照音声は 48kHz のみ（リサンプル非対応）

`sampleRate` 不一致は fail loudly。torchaudio の `sinc_interp_hann` 多相 FIR の移植は独立した
信号処理 1 本ぶんの実装・検証コストで、上流も 48kHz 入力ならスキップする。需要が出たら
ホスト純関数として独立に足す（黙った品質劣化ではなく明示拒否 — 横断不変条件）。

## Consequences

- テキスト →（参照 wav →）WAV の全経路が TS で閉じた。既定構成の生成実測: full ケース
  12.5 秒（S=161・6.44 秒の音声）/ voice-clone 14.0 秒（参照 7.6 秒 → 7.36 秒の音声）。
- **`sin` の引数が π を大きく超える経路の初観測**: |α·x| は decoder 側最大 15.9（≈5π）/
  encoder 側 71.3（≈22.7π）でも誤差は 5e-6 級 — GPU の `sin` 引数簡約が f32 精度を保つ
  ことの実測記録（`e2e_dacvae_test.ts` の tolerance docstring が正本）。
- speaker 門の tolerance は実 latent（実音声の決定的 encode）込みで確定（atol 7e-4）。
  DACVAE latent はほぼ単位分散で、W2 の合成標準正規が値域の性格を外していなかった。
- 未解決のまま残る制約（limitations 起票）: encoder 非タイル / 参照音声 48kHz のみ /
  `WAVE_FORMAT_EXTENSIBLE` 非受理 / 透かし非付与 / DiT の scores 136MB（ADR 0047 から継続）。
