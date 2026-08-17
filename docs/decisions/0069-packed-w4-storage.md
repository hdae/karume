# 0069: packed 4bit 格納（ADR 0019 の w4 棄却を reopen）

- Status: draft（2026-08-17 一括起草。Codex 漏れチェックと〔裁定 B / C〕の大域判断待ち）
- 関連: ADR [0019](0019-i8-weight-execution.md)（i8 経路 — 本 ADR が「w4 再測しない」を
  supersede し、実装規律は延長する）/ [0006](0006-quantization.md)（格納のみ量子化・
  fake-quant 正）/ [0063](0063-safetensors-physical-layout.md)（物理配置 — 本 ADR とセットで
  改訂）/ [0058](0058-numerics-opt-in-contract.md)（品質管理の分担）/
  [0066](0066-generation-context-state-slots.md)（検収モデル = LLM への波）
- 根拠:
  [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  §4（以下「調査 §n」）

## Context

検収モデル級（1〜2B param）を配布・常駐するには f16 でも 2〜4GiB 級で、4bit 格納が実用条件
になる。一方 ADR 0019 は「per-tensor / w4 group 量子化: 不採用確定。再測しない」と書くが、
この射程は過大（調査 §4.3）: 旧測定は **SBV2 voice 1 ファミリ・1 時点の int4 group 実測**
（2026-08-03）であり、親 ADR 0006 は group-wise を「低優先 backlog」としか書いていない。
棄却根拠の一つ「発話長の系統的短縮」は**採用済みの w8 でも観測されており**（ADR 0029:53）
w4 固有の基準として機能しない。WebGPU 上の 4bit は ORT MatMulNBits / MLC q4f16_1 /
llama.cpp WebGPU が実運用している（調査 §4.1）。

## Decision

### 1. ADR 0019 の supersede 範囲（reopen の正確な形）

- 「**再測しない**」を撤回する。旧測定の適用範囲を「SBV2 voice の音声 SNR / 発話長・
  per-tensor RTN および int4 group の 1 時点実測」に限定し、**テキスト生成（検収モデル）へ
  一般化しない**。数値の帰属も訂正する: −1.5〜+5.1dB は int4 group の実測（0019:17 が正・
  素朴 RTN へ帰属させた 0006:45 の側が不正確 — 調査 §7 wt LB-6 の検証済み補正）。
- 0019 のその他（i8 経路・±127 対称・平坦添字・タイル読み込み時 dequant・fake-quant 規律）
  は**本 ADR の土台としてそのまま有効**。
- 派生同期（実装波で）: ir-v1.md「group_size は実行経路が無い」・limitations の w4 項・
  container の groupSize 拒否。

### 2. 格納の表現 = bit 幅一般化（shape は論理のまま — 裁定 3a）

新しい格納 dtype **`storage.dtype: "i4"`** を追加する。**テンソル shape（IR `values{}` /
safetensors ヘッダ）は論理形のまま**で、バイト数だけ bit 幅から導出する
（GGUF / safetensors 本家と同配置 — 調査 §4.2。ORT 型の「物理 shape + 論理 attrs」は
消費側 op 契約〈linear の `W[out,in]` 等〉が総崩れになるため採らない）。

- safetensors リーダの改訂は **3 面を分離**する（第 3 巡で精密化 — `DTYPE_BYTES` の
  単純置換では済まない。現行表は「バイト長の計算」と「絶対 offset の整列検査」の両方に
  使われている）: ①**サイズ表（bit 単位）** — 検証は `numel × bits / 8` の厳密一致
  ②**整列表（byte 単位）** — I4 は要素整列の概念を持たず「テンソル先頭が 4 byte 整列」を
  要求（u32 束縛のため。F32/I32 = 4・F16 = 2 は従来どおり）③**view の型** — I4 に対応する
  TypedArray は存在しないため、view は **raw バイト（`Uint8Array`）+ 論理 numel の
  メタデータ**で表す（`STORAGE_ENCODING` の突合・container 検査・適格外の CPU nibble
  展開器も同時に改訂対象として列挙する）。
- **端数を作らない制約で整列問題を消す**: `i4` は量子化軸（最終次元 = linear の in 軸）の
  要素数が `group_size` で割り切れることを MUST とし、`group_size` は
  **2 冪かつ ≥ 16**（ORT と同制約 — 調査 §4.1。single-scale 例外は作らない）。これで
  行境界・group 境界が常にバイト整列し、テンソル総バイト数は 4 の倍数（u32 束縛の
  自然整列 — 末尾ゼロ詰め不要）。
- ADR 0063 改訂（セット）: 書き出し順は I8 の後ろ末尾に I4 を足す・リーダ / ライタ /
  `assert_reader_layout` を対で改訂。

### 3. 量子化形 = K 方向 group・対称 15 準位・scale f32〔裁定 B / C〕

- **group 対称量子化**: `scale = clamp(amax / 7, f32 tiny)`（group ごと — clamp は全ゼロ
  group の 0 除算を閉じる。ADR 0019 の `clamp(amax/127, f32 tiny)` と同文・第 3 巡で
  脱落を補正）・`q = clamp(round(w/scale), −7, +7)` を **offset 8 の unsigned nibble** で
  格納（値域 [1,15]・0 は未使用 = 15 準位）。amax 要素が厳密復元され fake-quant が冪等 —
  ADR 0019 の ±127 論証の 4bit 版（MLC q4f16_1 と同思想・調査 §4.1）。
- **scale の形を明文化**（第 3 巡の矛盾指摘の閉鎖 — ADR 0019 の「同 rank keepdim
  broadcast」は per-channel 専用で group には不適合）: scale の論理 shape は**重みと同 rank・
  量子化軸を group 数に置換**した形（linear `W[O,I]` → scale `[O, I/group_size]`）。
  container の scale 検査（現行 = keepdim broadcast 形）と WGSL の scale 束縛（現行 =
  出力チャネルあたり 1 値の `wscale[channel]` — weight-storage.ts:140-150）は **group 版の
  別分岐**として拡張する（添字 = `(row, k / group_size)`・タイル読み込み時に group 境界で
  引き直す）。CPU 展開経路も同じ添字式でビット一致を保つ。
- **zero-point 欄は作らない**〔裁定 B〕: 対称のみ。非対称（ORT の optional zero_points）が
  要る資産が実測で出たら本 ADR の改訂で欄を足す（欄の不存在が「語彙に無い」を表す流儀）。
- **scale は f32 companion**（既存 `storage.scale` + `storage.group_size` の解禁）〔裁定 C〕:
  ADR 0019 と同じく GPU dequant と CPU 展開のビット一致を保つ。f16 scale（MLC・GGUF）との
  トレードオフはサイズ表参照 — 帯域が問題になったら group_size を上げる側で調整する
  （bit 幅あたりのオーバヘッド: group 32 で f32 = +25% / f16 = +12.5%、group 128 で
  f32 = +6.25%）。既定 group_size は Phase 0 sweep（決定 6）で決める。

### 4. pack 順の明文固定 + 検出器

**「1 バイトに要素 2i（下位 nibble）/ 2i+1（上位 nibble）」**（ORT 型 — `unpack4xU8` 2 発の
展開順と対）。llama.cpp Q4_0 の split-half 順（byte j の下位 = 要素 j / 上位 = 要素 j+16）は
採らない — pack 順は上流でも割れている自由パラメータで、**間違えても形も型も合う沈黙誤値**
になる（調査 §4.1）。検出器 MUST: ①隣接要素が全て異なる非対称パターンのユニット
②group 長・行長が語境界と一致しない形（平坦添字の罠 — 0019 の同型検出器の 4bit 版）
③エクスポータ emit 直後の逆変換ビット一致門（`dequant(pack(w)) == fake_quant(w)`）。

### 5. 実行経路 = linear 限定で開始・タイル読み込み時 1 回展開

- 適格は **linear の重みスロットのみ**で開始する（LLM の支配項。0019 の全 5 op 展開は
  需要が出た op から追補 — embedding の w4 は語彙 262k 級で効果が大きいので最初の追補候補）。
  適格外の i4 宣言はロード時 CPU 展開（f32・GPU とビット一致 — 0019 と同じ受け皿）。
- WGSL は core builtin のみ（`unpack4xU8` + マスク / シフト — 調査 §4.1）・展開は
  **共有メモリタイルへの読み込み時 1 回**（0019 流儀・ORT の内積内展開は採らない）・
  語とレーンは平坦添字 MUST。
- scale 適用は要素ごと（`q·s` — 0019 決定と同一。縮約外形への変更は 0019 と同じ改訂条件）。

### 6. Phase 0 = Python fake-quant sweep が実装ゲート

runtime を触らない **format 候補 sweep** を先行する: 検収モデル（MiniCPM5-1B 先行 —
調査 §6.3）で group_size {32, 64, 128} × （必要なら）非対称の品質を fake-quant で比較し、
既定 group_size と〔裁定 B〕の実測根拠を得る。品質管理は ADR 0058 決定 5 の分担
（機械検証 = fake-quant 冪等・逆変換ビット一致 / 品質 = 人間レビュー + 助言的数値）。

## サイズ試算（Gemma 4 E2B 級・weight 2B param 概算）

| 形                              | bpw  | 概算      |
| ------------------------------- | ---- | --------- |
| f16                             | 16   | ~4.0 GiB  |
| i8 + per-channel f32 scale      | ~8.1 | ~2.0 GiB  |
| i4 group 128 + f32 scale        | 4.25 | ~1.06 GiB |
| i4 group 32 + f32 scale         | 5.0  | ~1.25 GiB |
| （参考）i4 group 32 + f16 scale | 4.5  | ~1.13 GiB |

## Consequences

- ADR 0019 は supersede 注記（範囲限定）を受ける。0006 の帰属記述は訂正注記。
- ADR 0063・ir-v1（storage dtype 語彙 + group_size 解禁）・container / safetensors
  リーダ・exporter emit / verify が実装波で対で動く。
- 受入条件: ①既存 f32/f16/i8 資産の読み書き・全 sha 門無風 ②逆変換ビット一致門
  ③pack 順検出器（決定 4 の 3 点）④GPU dequant と CPU 展開のビット一致 ⑤診断
  （`residentCompressedBytes` に i4 + scale が正しく載る）。
