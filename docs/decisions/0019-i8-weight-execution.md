# 0019 — i8（w8）格納の実行経路（per-channel scale + カーネル内 dequant）

- Status: accepted（2026-08-03）。**一部 supersede**（2026-08-18 — ADR
  [0069](0069-packed-w4-storage.md) 決定 1）: 「w4 不採用・再測しない」の射程を「SBV2 voice の
  音声 SNR / 発話長・per-tensor RTN および int4 group の 1 時点実測」に限定し、テキスト生成へ
  一般化しない。w4（格納 `i4`・K 方向 group）は 0069 で解禁・実装済み。i8 経路・±127 対称・
  平坦添字・タイル読み込み時 dequant・fake-quant 規律は 0069 の土台として現行のまま有効。
- 前提: ADR [0006](0006-quantization.md)（格納のみ量子化・fake-quant 正・bias/折り畳み定数は
  常に f32・FQN 突合・診断常設）/ [0018](0018-f16-weight-execution.md)（f16 実行経路 —
  weight 変種の生成部品と検証規律はここで確立済み）。プロトタイプには i8 の完全な先行実装と
  品質実測があり、本 ADR はその検証済み設計を Karume の構造（`WEIGHT_SLOTS` /
  `weight-storage.ts`）へ写像した上で、数値契約を 1 点だけ意図的に変える。

## 決定

- **方式: per-channel symmetric int8**。`scale = clamp(amax / 127, f32 tiny)`、
  `q = round(w/scale)` を **±127 に閉じる**（−128 不使用 — 最大絶対値要素が厳密復元され
  fake-quant が冪等になる）。zero-point なし。チャネル軸は出力チャネル:
  linear/conv1d/conv2d/embedding = 0、conv_transpose1d = **1**（転置レイアウト）。
  per-tensor はプロトタイプ実測（voice SNR 4.7〜5.7dB）で音声として不成立のため不採用。
  w4 も先行実測（SNR −1.5〜+5.1dB・発話長の系統的短縮）で不採用確定 — IR の `group_size` は
  予約のまま使わず、**付いていれば capability 不足で実行拒否**する（黙って per-channel と
  読むと group scale の意味が変わる沈黙誤値になる）。〔supersede — この w4 項は ADR 0069 が
  上書き: 実測の帰属は int4 group・射程は SBV2 voice の 1 時点。`group_size` は格納 `i4` で
  解禁済み（`i4` 以外に付いた場合の実行拒否は現行のまま）〕
- **scale は companion テンソル**（F32・weight と同 rank の keepdim broadcast 形）。IR v1 の
  `storage.scale`（既存語彙）で**明示宣言**する — キー命名は自由だが実テンソル名との衝突を
  emit 時に検査する。ロード時に存在・dtype・broadcast 可能形を検証して fail loudly。
- **GPU 常駐 + カーネル内 dequant**: ペイロードを `array<u32>` で束縛し
  **`unpack4xI8`（core WGSL・feature 依存ゼロ — ADR 0002/0018 と整合）**で展開する。
  MUST: 語と位置の割り出しは**平坦添字**（`unpack4xI8(w[i >> 2u])[i & 3u]`）から作る —
  行内相対添字は行長が 4 の倍数のとき偶然一致する（f16 の偶奇と同型の罠。検出器は
  **行長が 4 の倍数でないテスト**）。末尾は 4 バイト整列までゼロ詰め（値に影響しない）。
- **scale の適用は要素ごと（読み出し時 dequant）**: `out = Σ x·(q·s) + bias` の形。
  GPU の dequant と CPU 展開（適格外の f32 展開経路）が**ビット一致**し、ADR 0018 の検証
  規律（ユニットで固定・E2E は実装誤差だけを見る）を i8 でも維持できる。プロトタイプの
  縮約外形 `(Σ x·q)·s` は乗算が出力要素あたり 1 回で済むが 1ulp 級の不一致が原理的に生じ、
  検証が tolerance 論に落ちる。コスト: linear は共有メモリタイルへの読み込み時 1 回
  （MAC あたりではない）、conv 系は `s` がループ不変なので巻き上げで乗算 1 個 —
  現行素朴カーネルの支配項ではない。perf で縮約外形や DP4a（整数 MAC・プロトタイプ実測
  4.73×）へ切り替える場合は**本 ADR の改訂 + tolerance 再導出を条件**とする。
- **適格 = `WEIGHT_SLOTS` 全 5 op**（linear / conv1d / conv2d / conv_transpose1d /
  embedding）。プロトタイプは 3 op（conv2d・embedding なし）だったが、カーネルごと手書き
  だった構造的理由は `weight-storage.ts` 共有で消えている。embedding は縮約が無く per-row
  dequant のみでビット一致が自明。適格判定は `eligibleCompressedInitializers` を**そのまま
  共用**（新設なし — bias を表に載せない構造は元々 i8 由来であり降格系バグの余地なし）。
  適格外の i8 宣言はロード時 CPU 展開（`q·s` の f32 丸め 1 回 = GPU と同値）。
- **`weight-storage.ts` の拡張は 3 点に閉じる**: ① `WeightStorage` に `"i8"`
  （キー判別子 `:wi8`・`WEIGHT_STORAGES` 網羅でスナップショット/縮退ハーネスが機械的に回る）
  ② scale の**追加束縛**宣言（bind group layout が変種で変わる — 束縛宣言の生成も
  weight-storage.ts に閉じ込め、executor の bind entries と対で分岐）③ `weightRead` に
  **出力チャネルの scale 式**引数（f32/f16 は無視 — **両変種の生成物バイト不変 MUST を
  維持**。当初案の「チャネル添字」渡しは scale の読み出しが重み要素ごとになり、conv 系の
  ループ不変巻き上げをソースに書けないため、束縛済み局所変数の式を渡す形へ改訂
  〈2026-08-03・運ぶ情報は同一・束縛忘れは WGSL コンパイルエラー〉）。
  scale バッファのバイト数は `residentCompressedBytes` 側に加算（診断常設の一貫性）。
- **エクスポータ**: fake-quant は**実効重み**に当てる（weight_norm 除去・LoRA 焼き込み・
  conv3d→conv2d スライスの**後**、参照採取の**前** MUST — 使われない要素が amax に効くと
  scale がずれる）。突合は FQN + 期待本数照合（`id()` 禁止 — ADR 0006）で、i8 指定なのに
  0 本なら fail loudly。emit は fake-quant が使った scale を**そのまま**書き（再計算は f32
  丸めで golden 対応が壊れる）、格納時に逆変換ビット一致 `torch.equal(q8.to(f32)·s, t)` を
  門にする。1 ターゲット 1 dtype（混成格納は将来の拡張）。I8 は 1 バイト要素で reader の
  整列制約が無いため並び順は**末尾**（既存 F16 規則の後ろ）。
- **資産系列**: `models/anima-i8/` は **transformer のみ**（text/cond/vae は f16 系列と共有 —
  DiT の −1.87GiB が支配項で、VAE の i8 化はプロトタイプ実測 −24.6MB と桁が 2 つ違う）。
  conv2d/embedding の wi8 は tiny golden + ユニットで被覆する。サイズ試算: DiT フル 28 層
  f32 7,465MiB → f16 ≈3,733MiB → **i8 ≈1,867MiB**（scale オーバヘッド実測 0.4〜0.9%）。
- **品質ゲート**: fake-quant 方法論により E2E は実装誤差しか測らない — **量子化の質は別軸**で
  測る。①DiT/VAE 直交分解の計測台本を移植（denoise 2 回 + decode 4 条件で PSNR/relRMS）
  ②デモ画像の w8 変種を生成して**目視裁定**（プロトタイプ先行事例: DiT のみ i8 で
  PSNR 27.28dB — 拡散は軌道が変わるため数値は「劣化」と「別の絵」を区別しない — 目視で採用）。

## 検討した代替案

- 縮約の外で scale（プロトタイプ形）: 乗算最少だが GPU/CPU 展開のビット一致を失う。
  perf マイルストーンの upgrade path として温存（上記の改訂条件付き）。却下（保留）。
- per-tensor / w4 group 量子化: プロトタイプの数値・聴感ゲートで不採用確定。再測しない。
  〔supersede — 「再測しない」は ADR 0069 決定 1 が撤回。w4 group は再測（Phase 0 sweep）の
  うえ格納 `i4` として解禁済み。per-tensor の不採用は現行のまま〕
- DP4a（`dot4I8Packed` 系の整数 MAC）: プロトタイプ実測 4.73× と大きいが計算契約ごと変わる。
  perf で裁定。
- 全ターゲット一括 i8 系列: VAE/text 層の削減量が小さく検証コストだけ増える。
  transformer 単独 + tiny golden 被覆で足りる。却下。
