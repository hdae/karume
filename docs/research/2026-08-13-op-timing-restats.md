# op 別 GPU 内訳の再実測 — perf-ledger 第 1 段の消化と K-8 の裁定

> NOTE: 時点スナップショット。数値は 2026-08-13 の実測（RTX 3080 Ti / driver 610.43.03 /
> Linux / Vulkan / Deno 2.9.4・main `01bffe6`）。手法は
> [2026-08-10-op-timing-stats.md](2026-08-10-op-timing-stats.md) §5 の再現（差分は §5）。
> 前回比の分母が違う（当時は壁 13.9s 時点）ことに注意 — 比率の直接比較は §1.3。

発端は perf-ledger 第 1 段「現行 HEAD での op 別 GPU 内訳の再取得」（2026-08-12 裁定）。
量子化波で未消化のまま残っていた Irodori DiT の内訳初取得と、K-8（attention PV i8a8 の
Vᵀ + quantize_rows 融合）の採否根拠の取得を兼ねる。

**結論を先に**:

1. **K-8 は棄却**（§2）。対象プール（Vᵀ permute + V 量子化）は実測 **約 79ms/生成 =
   GPU の 1.0%**。融合で消せるのはその ~61%（vt の書き + 読み）= **壁の ≈0.45%** で、
   新カーネル + byte parity 門の実装・保守コストに見合わない。全削除の理論上限でも
   壁 0.73%。
2. anima w8a8-1024 の序列は前回から不変: **DiT linear i8a8 45.0% / DiT attention 3 種
   20.7% / VAE 16.3%**（全 GPU 比・§1）。カーネル最適化の桁は引き続きこの 3 群。
3. **Irodori（w8a8 既定・voice-clone）の内訳を初取得**（§3): **dit 51.6% /
   codec-decoder 38.7% / codec-encoder 9.2%**。codec-decoder の中身は
   `conv_transpose1d` **4 dispatch で 1,707ms**（427ms/dispatch）+ conv1d 27 dispatch で
   1,108ms — **convT 4 本だけで全 GPU の 23%**。DiT の masked attention は融合カーネルで
   なく bmm 分解経路（`bmm:v2:f32` 12.9% + safe_softmax 2.2%）で走っている（K-5 の領域）。
4. Irodori は **anima と逆にホスト側が壁の主因**（§3.3）: 壁 16.1s（計測込み）− ロード
   1.3s − GPU 7.4s = 露出 7.4s。dispatch 161,639 本（anima の 6.4 倍）に対する計測装置代
   （≈2s 推定）を引いても露出ホストは ≈5s。H-5/H-1 系の裏付け。
5. 再現性と経路同一性: A/B 差は anima ±0.2% / irodori ±0.7%・PNG 門 2 構成とも参照一致・
   irodori WAV sha256 は 3 走完全一致・clampedNegativeSamples 全走 0。

## 1. anima の再実測

### 1.1 段別（GPU timestamp の ns 合計・A 走。B 走との差は表 Δ）

**w8a8-s16 / 1024×1024 / 8 step / seed 42 / guidance 1（manifest 既定）** — 壁 10.42s
（計測込み・装置代 ≈0.33s）/ GPU 7,930.57ms / PNG sha256 参照一致:

| 段               | run |     GPU (ms) | dispatch | 全体比 |  Δ B/A |
| ---------------- | --: | -----------: | -------: | -----: | -----: |
| text_encoder     |   1 |        33.22 |      901 |   0.4% | −0.05% |
| text_conditioner |   1 |        10.05 |      469 |   0.1% | −0.05% |
| transformer(DiT) |   8 |     6,594.17 |   21,216 |  83.2% | −0.19% |
| vae_decoder      |   9 |     1,293.13 |    2,727 |  16.3% | +0.04% |
| **合計**         |  19 | **7,930.57** |   25,313 |   100% | −0.15% |

**f16 / 1024**: GPU 16,755.56ms（transformer 91.9% / VAE 7.9%）・壁 19.5s（計測込み）・
PNG 参照一致。DiT 内は linear wf16 67.3% / attention qk+pv 22.0%。

### 1.2 DiT の op キー別（w8a8・8 step 合計・上位のみ）

| ms       | DiT 比 | dispatch | キー                                                     |
| -------- | -----: | -------: | -------------------------------------------------------- |
| 3,567.21 | 54.10% |    3,632 | `linear:v4:i8a8:tile128x64r8x8w8x16k16v4:dp4a`           |
| 751.20   | 11.39% |      448 | `attention_pv:v3:i8a8:tile64x128r8x8w16x8k16v4:dp4a:s16` |
| 577.76   |  8.76% |      448 | `attention_qk:v3:i8a8:tile128x64r8x8w8x16k16v4:dp4a:s16` |
| 447.13   |  6.78% |    4,976 | `quantize_rows:v1:f32>i8:pertoken:wg256`                 |
| 341.34   |  5.18% |    6,968 | `strided:v1:f32:r4:wg256`                                |
| 310.57   |  4.70% |      448 | `attention_stats`（rc16 + rc2 の 2 キー合算）            |
| 225.56   |  3.42% |      904 | `rms_norm:v1:f32:lastdim:wg256`                          |
| 87.90    |  1.33% |      680 | `adaln_norm:v1:lastdim:f32:wg256`                        |

キー語彙は 2026-08-10 時点から移動している（tile 幾何判別子化・attention_stats の
rc16/rc2 分割・`adaln_norm` 新設 + 融合カウンタ adaln 680）— 前回表とキー名で突合しない。

### 1.3 前回（2026-08-10・壁 13.9s 時点）との比較

全 GPU 比で: linear i8a8 40.0% → **45.0%** / attention 3 種 23.2% → **20.7%** / VAE
19.1% → **16.3%**。分母が縮んだ（11.37 → 7.93s）ぶん相対の並び替えはあるが、
**「DiT linear + attention と VAE conv2d が 8 割強」という結論は不変**（82.4% → 82.0%）。
perf-ledger K-1 の失効注記はこの値で更新した。

## 2. K-8（attention PV i8a8 の Vᵀ + quantize_rows 融合）の裁定 — 棄却

対象は `#buildAttentionPvI8a8`（executor.ts）の (a) strided Vᵀ permute（f32 実体化）→
(b) quantize_rows → (c) PV GEMM のうち (a)+(b) の融合。対象コストの実測按分:

- **Vᵀ permute ≈ 38.4ms/生成（f16 差分法）**: f16 の DiT `strided` は 6,520 dispatch /
  302.91ms、w8a8 は 6,968 / 341.34ms。**差の 448 dispatch はちょうど PV の Vᵀ permute**
  （f16 経路には存在しない・他の strided 用途は両系列で同一形）なので、差 38.43ms が
  Vᵀ の実コスト。
- **V 量子化 ≈ 40ms/生成（dispatch 按分）**: quantize_rows 4,976 dispatch のうち V 向けは
  448 本（残りは linear 活性 3,632 + q/k 896）。一様按分で 447.13 × 448/4976 ≈ 40ms。
  要素数モデル（V 総量 ≈2.1G 要素 / 全量子化 ≈24.8G 要素）でも ≈38ms — 2 法一致。

**対象プール ≈ 79ms = 全 GPU の 1.0% = 壁の ≈0.76%**。融合で消えるのは vt（f32）の
書き + 読み（対象トラフィック 27GB 中 17GB ≈ 61%）で、**期待利得 ≈ 48ms ≈ 壁の 0.45%**。
(c) の PV GEMM 751ms には触れない。amax→scale→丸めの順序一致で byte parity 門が組める
という設計上の見立て自体は正しいままだが、新カーネル + 門の追加保守に対して利得が
1 桁足りない。**台帳の「性能は要実測」条項どおり棄却**（K-3 と同じ「採らないが記録」へ）。

## 3. Irodori（w8a8 既定）の内訳 — 初取得

**voice-clone ケース**（WAV 門の同ケース準拠: seed 1234 / steps 40 / cfg 3,5,3 /
S=170 / forwards 100 / 出力 6.8s）。壁 16.08s（計測込み）/ GPU 7,364.41ms /
WAV sha256 は 3 走完全一致（w8a8 席に固定 sha 門は無い — 安定性確認として記録）。

### 3.1 段別

| 段            | run |     GPU (ms) | dispatch | 全体比 |
| ------------- | --: | -----------: | -------: | -----: |
| backbone ほか |   6 |        41.88 |    1,660 |   0.6% |
| codec-encoder |   1 |       674.98 |      189 |   9.2% |
| dit           | 100 |     3,797.67 |  159,600 |  51.6% |
| codec-decoder |   1 |     2,849.88 |      190 |  38.7% |
| **合計**      | 108 | **7,364.41** |  161,639 |   100% |

### 3.2 支配 op

- **dit**: linear i8a8 58.5% / **bmm f32 12.9%（2,400 dispatch）+ safe_softmax 2.2%** /
  rms_norm 6.0% / quantize_rows 4.7% / strided_write 4.2%。masked attention は融合
  attention でなく **bmm 分解 + safe_softmax の素経路**（S=750 級の scores 実体化を
  毎 forward 実走 — ADR 0047/0044 の by-design。オンライン化は K-5）。attention 系
  （bmm + softmax + 周辺 permute の一部）で dit の ≈15〜20%。
- **codec-decoder**: `conv_transpose1d:v1:f32:gather:wg256:wi8` が **4 dispatch で
  1,707.1ms（60.0%・427ms/dispatch）**・conv1d direct が 27 dispatch で 1,107.8ms
  （38.9%）。**convT 4 本 = irodori 全 GPU の 23.2%** — 単発粒度としては全パイプライン
  最大（SBV2 の同カーネルは 13.7ms/dispatch だった — 2026-08-10 §3。2 桁大きい）。
- **codec-encoder**（voice-clone のみ実行）: conv1d 96.6%。

### 3.3 壁とホスト（読み方の注意つき）

壁 16.08s − ロード 1.31s − GPU 7.36s = **露出 7.4s**。ただし本走は gpuTiming ON で、
装置代は dispatch 数比例（anima w8a8 の実測 0.33s / 25,313 本）— 161,639 本に外挿すると
≈2.1s が計測起因。**補正後の露出ホスト ≈5.3s ≈ 壁（補正後 ≈14s）の 4 割**で、
anima（露出 gap ≈1s 台）と逆にホスト律速側。dispatch 密度（1,596 本/forward × 100 +
codec）が主因で、H-5（反復状態 GPU 常駐）/ H-1（二段待ち解消）/ K-7（融合受理拡張 =
dispatch 削減）の裏付け実測になる。無計測の壁は本波では取っていない（速度裁定に使う
場合は素走を別途取ること）。

## 4. 付随して確定した事実

- gpuTiming ON でも PNG 門 2 構成の sha256 参照一致・irodori WAV sha256 3 走一致 —
  計測経路の無害性を実運用経路で再追認。clampedNegativeSamples 全走 0。
- f16 の DiT strided 6,520 dispatch と w8a8 の 6,968 の差がちょうど PV Vᵀ の 448 本 —
  「w8a8 と f16 のグラフ構造は attention permute + quantize を除き同一」という前回観察の
  現行 HEAD での再確認（§2 の差分法の前提）。
- i8a8 attention の効果の現在値: qk+pv 合算で f16 3,396ms → w8a8 1,329ms（2.56 倍）。

## 5. 追記（同日・K-10 の着地 — convT residue grouping の検収）

§3.2 の発見を受けて K-10 を実装した（`d08dd8a` — conv_transpose1d の縮約を residue
grouping へ・キー v2）。検収 A/B（同ケース・クールダウン規約・§5 と同じドライバ）:

- **convT: 1,707.1 → 189.8 / 190.2ms（9.0 倍）** — stride 8 の理論削減 8 倍 + 分岐除去分。
- codec-decoder 段: 2,849.9 → 1,331.4ms（2.14 倍）。残りは conv1d direct 1,108ms が支配
  （K-4 の implicit GEMM 案の対象 — 本波の外）。
- **irodori 全 GPU: 7,364 → 5,818 / 5,791ms（−21.2%）・壁 16.08 → 14.52 / 14.50s
  （計測込み ×1.11）**。GPU 内の新比率: dit 65% / codec-decoder 23% / encoder 11%。
- ビット同一の証明: v1/v2 tap 列総当たり 573,839 ケース不一致 0（u32/i32 意味論模擬）・
  verify 1045/0（実 GPU — SBV2/irodori WAV sha256 門・dacvae タイル Uint32 門込み）・
  検収走の WAV sha256 が変更前と**同一 digest**。

## 6. 手法（前回 §5 との差分）

- 段の帰属は `Session.prototype.run` のラップでなく **`onRunDiagnostics`**（パイプライン
  公式席 — 前回 §6 の「観測面の恒久化」は解消済みで、e2e 門も同席を使う）。読み取り点は
  §5 と同一（run 解決直後・次 run の resetTiming 前）。ラベルはパイプライン宣言由来。
- irodori は WAV 門の voice-clone ケース準拠（w8a8 席へ差し替え）。8 コンポーネント中
  7 つ + 3 分岐 CFG を通る構成として選択。素の TTS（参照なし）は codec-encoder が
  落ちる以外同型。
- 壁時計は計測込みのみ（装置代 anima ≈41ms/step・irodori は §3.3 の外挿注意）。
- クールダウン 40°C 以下・A/B 各 2 走・台本と生データ JSON は scratchpad（揮発）—
  本ドキュメントの表が写し。
