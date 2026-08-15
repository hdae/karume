# 画像デモ / w8 / perf — recon と実装波の記録（2026-08-03）

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。

> recon 5 レッグ（トークナイザ / デモパイプライン / Turbo LoRA / i8 / perf 技法）と、
> 画像デモ実装波（D1 Turbo 資産 / D2 トークナイザ / D3 デモ本体）の要点。設計裁定の正本は
> ADR [0019](../decisions/0019-i8-weight-execution.md)（w8）、デモの使い方と実測は
> 当時の `examples/anima/README.md`（現在は `packages/models/src/anima/` へ移設済み）、
> トークナイザの機構は `tools/exporter/karume/anima_text.py` の docstring。ここには**他に置き場の無い
> 恒久情報**（perf 材料・recon の反証記録・隣接観測）だけを残す。

## perf 材料（プロトタイプ実測の棚卸し — perf マイルストーンの入力）

GEMM（M=4096, N=K=2048, Blackwell, 20 反復・プロトタイプ実測）:

| 変種                                                                          |    GFLOP/s |       対 16×16 単純タイル |
| ----------------------------------------------------------------------------- | ---------: | ------------------------: |
| 16×16 単純タイル（Karume の現行 matmul 相当）                                 |      3,506 |                     1.00× |
| レジスタブロッキング 4×4（64×64 タイル・256 スレッド・K タイル 16・共有 8KB） |      8,754 |                     2.50× |
| レジスタ 8×8（4×4 比で劣化 — レジスタ溢れ・不採用）                           |      3,867 |                     1.10× |
| **レジスタ 4×4 + vec4（K/N%4==0 条件）**                                      | **10,754** |                 **3.07×** |
| f16 計算（shader-f16・`enable f16`）                                          |      5,057 | 1.44× — 「本命ではない」※ |
| DP4a（i8 整数 MAC）                                                           |     16,599 |                     4.73× |

> ※訂正（2026-08-03・#29 recon）: **f16 の 1.44× は設計判断に使えない数値** — 基準が旧
> 16×16 カーネル（現行 4×4+vec4 比 0.47×）で、ベンチ実体に交絡 4 点（vec4 読み無し /
> MAC 毎 f16→f32 変換 / 入力が f16 subnormal / 検算無し）。詳細は
> [shader-f16-recon](2026-08-03-shader-f16-recon.md) §2。DP4a の 4.73× の基準問題は
> ADR 0025 で訂正済み（現行比 1.543×）。

- 内側ループは B の vec4 を 1 回ロードして 4 出力 × 4 成分の FMA に使い回す構造。
- flash attention（T3b: 32×16・T128・1 スレッド 2×2）は 5,498 GFLOP/s。共有メモリの
  バンク衝突などデータ配置 3 点の修正で、ヘッド次元 D≤144 制約下 +12.5% まで縮小した経緯あり。
- **subgroup-matrix 系は WebGPU に無く、テンソルコアへは届かない**（プロトタイプ recon の結論）。
- conv1d / conv2d はプロトタイプ側も素朴実装のまま（「VAE decode が横ばいなのはこれ」）。
- timestamp-query は GFLOP/s 計測に使用実績あり。subgroups は feature 検出のみで未使用。
- Karume 側の既存 wall-clock: DeBERTa 24 層 run ~230ms / SBV2 front run2 90–99ms /
  **Anima デモ（512px・実測 2026-08-03）: turbo 61.4s（4,893ms/step）・base 379.1s
  （11,446ms/step）**。DiT 単発 forward ≈ 4.9–5.7s が現在の支配項。

## w8 recon の補足（ADR 0019 に載せなかった実測）

- プロトタイプの品質実測: int8 per-channel = voice SNR 12.8〜24.2dB（採用）/
  per-tensor = 4.7〜5.7dB（却下）/ int4 group = −1.5〜+5.1dB + **発話長の系統的短縮**
  （SNR と別の劣化軸）で不採用。DeBERTa hs[-3] relRMS 2.3〜4.1%。
- Anima 直交分解: VAE のみ i8 → PSNR 40.66dB / DiT のみ → 27.28dB（「劣化」と「別の絵」を
  区別しない指標 — 目視で採用）。VAE の i8 化は 49.4→24.8MB にしかならない。
- 実装後回帰（プロトタイプ）: verify（CPU dequant 先行）で voice 全 5 ケース**ビット一致**・
  DeBERTa 4/4 ビット一致。i8 は f16 降格バグを「bias を重みスロットに載せない」構造で
  最初から回避していた（Karume ADR 0006 はその一般化）。

## recon の反証記録（後続が旧情報を掴まないための正誤表）

1. **「`(?i:)` は ASCII 相当」（recon L1）→ 誤り**。Rust の `(?i:)` は Unicode simple case
   folding で U+017F（ſ）≡ s。1 文字探り針の検査では素通りする（波D2 が実測で穴を踏んで
   から 11 文脈 12,232,704 件に拡張して封鎖）。正本: `anima_text.build_case_fold`。
2. **「long_s ケースが大小無視の境界を叩く」（recon L1）→ 誤り**。アポストロフィが無く
   短縮形分岐に入らない。境界を叩くのは `apostrophe_fold`（それも id 列には出ない —
   byte-level BPE が結合し直す。検出は分割直叩きテストのみ）。
3. **「metaspace ケースが MergedWithNext 分岐を叩く」（recon L1）→ 誤り**。現行の正規化表は
   U+2581→U+0020 なので encode 経路では到達不能（表が変われば到達しうるため実装は正本忠実の
   まま・直叩きテストで固定）。
4. **「反復で誤差は積み上がらない」（2 step チェーンの観測）→ 外挿不能**。turbo 10 step
   完走では約 1.75×/step で単調累積（9 step で 160 倍）。正本:
   `tests/e2e_anima_turbo_test.ts` の実測表。

## 隣接観測（未調査・タスク未着手）

- **per-step 時間が走行中に伸びる → 解決済み（2026-08-03・波T2 実測）: GPU クロック低下
  （熱/電力）が全量**。時間比とクロック比が 4 走行とも一致・ms×MHz は 4〜8% しか動かず・
  r(step 時間, clocks.sm) = −0.92〜−0.99・冷機 45℃ 対照で Σ(ms×MHz) 差 0.1%・PNG は熱状態を
  跨いで sha256 一致。submit 回数の増加（197→534）は msPerWorkgroup がクロック比で伸びた
  **結果**で、dispatchCount は 3,137 一定 — 適応制御・プール側の疑いは否定。正本 =
  examples/anima/README.md「per-step 時間の伸び」節。perf での含意: 壁時計の比較は熱状態を
  揃えるか ms×MHz で正規化する。
- **DiT ロード中の nvidia-smi ピークが常駐重みの約 2 倍**（7,875MiB vs 3,733MiB）。
  アップロード経路の一時領域。ピーク削減も perf/メモリ側の材料。
- `timestepsProj` は SLEEF との丸め差で参照とビット一致しない（最大 2 ULP・5.96e-8）。
  SLEEF 写経は**やらない**と裁定済み（複雑さが見合わない — `host/sampler.ts` の実測表）。
- トークナイザ表の共通化候補: `examples/anima/text/code_ranges.ts` と
  `examples/sbv2/assets.ts` の区間表が同型（統合は別タスク）。`host/random.ts` も同様。
- **SBV2 デモの NFC エンジン差（未対処・裁定 = 実用段階まで現状維持〈2026-08-03
  ユーザー〉）**: `examples/sbv2/text/tokenizer.ts` が素の `normalize("NFC")` を使っており、
  Anima 側で実測・根治した「正本（Rust `unicode-normalization`）と ICU が 123 cp で割れる」
  問題を同じ形で踏みうる。対処するなら Anima の分節表方式（`normalizeNfc` /
  `build_nfc_segments` — コミット 4bba9c4）をそのまま移す。DeBERTa の normalizer 構成が
  NFC かの確認から（違えばこの懸念自体が消える）。
- latents_mean/std を exporter が emit する案（現状はデモ側の手写し定数 + フィクスチャ 3 系列
  とのビット一致テストで固定）。
