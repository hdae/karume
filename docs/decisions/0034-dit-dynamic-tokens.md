# 0034: DiT の S 化（トークン長 1 シンボルの動的グラフ・--dit-graph dyn）

- Status: accepted（主門 = 静的グラフとの Uint32 完全一致が 3 ケース成立 —
  絵が 1 ビットも変わらないため目視ゲートは不要）
- Date: 2026-08-04
- 適用範囲の注記: 「static / dyn の併存・static 既定」は**エクスポータ開発時のルール**として
  現行（固定形は検証アンカーとして残る）。**配布資産と models パイプラインは ADR
  [0038](0038-manifest-v1.md) §4 により S 形のみ**で、`--dit-graph` ノブは存在しない。
- 関連: recon = [dynres-vae-tiling](../research/2026-08-03-dynres-vae-tiling.md) §2.2
  （S 案の初出・障害物の列挙）/ ADR 0016（解像度焼き込み — 本 ADR はその**追加系列**）/
  0010（記号次元の定数畳み込み要件 — Dim("S") の上限根拠）/ 0013（表の入力昇格の先例 =
  SBV2 相対位置表）/ 0033（VAE 側の解像度非依存化 — 両輪で「1 資産セットの可変解像度」）

## Context

DiT の解像度依存は実測で **3 定数だけ**（padding channel `[1,1,H,W]` のゼロ・rope の
cos/sin 表 `[1,1,S,128]` ×2）。IR の次元言語（1 次元 1 シンボルの一次式）は `H·W` を
書けないが、グラフの入口を patchify の**後ろ**へずらせばトークン長 `S` の 1 シンボルで
全ノードが書ける（recon §2.2）。ランタイムは空間 symbolic を追加改修ゼロで受けられる
ことも recon で確定済み（plan は run ごと・キーに shape なし）。

## Decision

1. **追加系列であって置換ではない**: `export_anima.py --dit-graph dyn`（transformer 専用）が
   `models/anima-turbo-{f16,i8}-dyn/transformer` を emit。既存の静的資産・E2E・tolerance・
   スナップショットは 1 バイトも動かさない。デモは `--dit-graph {static,dyn}`（既定
   static）。
2. **グラフ入口は patchify 後**: 入力 `tokens [1,S,68]`（68 = 17·2·2・恒常ゼロの padding
   channel 込み）+ `rope_cos` / `rope_sin [1,1,S,128]`（**入力昇格** — SBV2 相対位置表と
   同じ手）。出力は unpatchify 前のトークン。patchify / unpatchify / rope 構築はホスト
   （examples/anima/host/dit-tokens.ts・純関数）。**S 依存の焼き込み定数はゼロ**
   （rank≥3 initializer 0 本を E2E が固定・静的形は同じ検査で 3 本 = 非恒真）。
3. **rope 表は「計算」ではなく「軸別素表の並べ替え」**（実装波の設計変更を裁定で受理）。
   torch の f32 cos/sin は正しい丸めと 1 ulp ずれる（実測: 定義域 8,192 通りで cos 472 /
   sin 231 件）が JS の Math.cos は正しく丸まるため、**TS で式を写す限りビット同一は
   原理的に成立しない**。エクスポータが `model.rope` の出力から軸別素表
   （`rope_base.safetensors`・66KB・解像度非依存・行数 128 = latent 256 = 2048px 相当の
   モデル側天井）を切り出し、ホストは `[t,h,w,t,h,w]` に並べるだけ（三角関数を 1 度も
   呼ばない — トークナイザの「表に焼く」と同じ判断）。
4. **主門 = S 形 ≡ 静的グラフの Uint32 完全一致（実 GPU）**。ホスト段は純置換 / 表の写し
   なので tolerance ではなくビット同一が期待値 — 破れたら設計前提の破れ（緩め禁止）。
   `Dim("S", min=2, max=16384)`。**torch.export は S 依存 guard を 1 つも作らなかった**
   （recon §2.2 障害物 1 は否定）。
5. `--verify` は「ホスト patchify → S 形 → ホスト unpatchify ≡ パッチ前 diffusers」を
   ケース別 adapter（unpatchify）越しに突合（bit_exact 2 ケース）。golden の 2 点評価は
   **解像度 × timestep の両方**を変える（S=1,024 / 4,096 — 同じ S を 2 本並べると束縛の
   失効が数に出ない）。

## 実測（2026-08-04）

- **主門 3 ケース全て Uint32 完全一致**: 512px f16 / 1024px i8 / 512px i8 + w8a8 +
  attention a8 + s16（量子化ノブ全部入り・パイプラインキー集合も一致）。
- `--verify` 2 ケース bit_exact=True（maxdiff 0.000e+00）。rope 表のホスト再構成は
  512/1024 の静的資産焼き込み値と**バイト一致**。
- デモ PNG sha256 が静的経路と**完全一致**（512px turbo f16 同一 seed）。
  per-step 静的 1,046ms / S 形 1,051ms（測定限界以下）・**dispatch 2,633 → 2,623**
  （patchify/unpatchify の permute −10）。
- emit: f16-dyn 3.91GB / i8-dyn 1.96GB（各 81s・initializer は静的形からちょうど 3 本減）。
- 既定経路の PNG 門 2 系列とも実行前後で完全一致（メイン実測・predict 分岐リファクタ込みでバイト不変）。
- verify 697/0/4・pytest 1,922（メイン自己実測）。故障注入 7 件（赤 6 + 検出限界 1）。

## 検出限界・知見（本タスクの新記録）

1. **rope 素表の h↔w 取り違えは数値網で原理的に検出不能** — Anima の `rope_scale` が
   `[1.0, 4.0, 4.0]`（h と w が同値）のため `cos_h == cos_w` がバイト単位で成立する。
   「実測グラフの permute は全て対合」（Pitfalls）と同型の**モデル構成依存**の罠。
   実働の検出器は「素表を上流出力から切り出す」実装形そのもの（写し間違いの余地がない）。
2. **torch の f32 三角関数は正しい丸めではない**（SLEEF・1.0 ULP 保証）— 「参照と同じ式を
   書けばビット一致する」は超越関数では成立しない。表に焼くのが唯一の恒等手段。
3. TypedArray を assertEquals に渡すと赤のとき差分整形が実質終わらない（15 分打ち切り
   実測）— 大型配列の突合は「最初の食い違い + 件数」形の assert（assertSameBits）で。

## Consequences

- **DiT は 1 資産で全解像度**（素表天井 = 2048px 相当まで）。ADR 0033 の VAE タイルと
  合わせ「1 資産セットの可変解像度」の部品が揃った — 残るはデモの任意解像度・非正方形
  配線（タスク #23・main.ts の prepareDynDit だけが正方前提。host 側は H≠W 対応済み）。
- 2048px の実生成には DiT attention の中間 S（S=16,384 で 16×16384²・s16 でも 8.6GiB）が
  載らない問題が別途残る（メモリの工事 — 保留枠）。
- 静的系列は当面併存（既定経路・PNG 門の正本）。統合の判断はモデル配布形式の設計
  （HF リポジトリ形式 — 保留中）と合流させる。
- models/anima-turbo-f16-1024 は削除済みのため 1024px の主門対は i8 系列で張った
  （解像度の網は 512/1024 で閉・dtype の網は 512px 対で閉）。f16×1024 の対が要る時は
  再 emit（3.7GB・3 分）で張れる。
