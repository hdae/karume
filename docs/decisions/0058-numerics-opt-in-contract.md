# 0058 — numerics opt-in の一般契約（個別 named option・既定 = 参照経路）

- Status: accepted（2026-08-14・ユーザー承認「このまま承認できます」。骨子 = 案 a〈個別
  named option〉と「一括フラグは将来課題」は 2026-08-13 の先行裁定）
- 対象: runtime の `SessionOptions` 実行変種面。既存 3 軸（`linearCompute` /
  `attentionCompute` / `attentionScoreStorage`）の流儀を契約へ昇格し、以後の「数値契約を
  割る最適化」（直近 = perf-ledger K-5 / K-2）が従う席の作り方を 1 本に固定する。
  配布資産側の量子化系列（quant 席 — [ADR 0050](0050-irodori-quant-series.md)）は
  **格納形の軸**で本 ADR の対象外（交差の規約のみ決定 7）。
- 背景方針: ビット同一門は「実装が正しいことの指標であって目的ではない」
  （ユーザー方針 2026-08-13 — 正本は [perf-ledger](../perf-ledger.md) ヘッダ）。

## 決定

1. **数値を変える最適化は、1 最適化 = 1 個の named option として `SessionOptions` に席を
   持つ（案 a）**。型は string literal union で、参照経路を表す値（`"f32"` 等）が既定。
   命名は `<対象><軸>` の既存流儀（`linearCompute` と同型）。boolean は採らない —
   同一対象に第 2 変種が来た時に軸ごと作り直しになる（`attentionCompute` が f16 の後から
   i8a8 を受けた実績が根拠）。
2. **既定 = 参照経路 MUST**（[ADR 0028](0028-f16-compute-variants.md) 決定 1「auto 禁止」の
   一般化）。アダプタ能力・資産の格納形・グラフの形から数値を変える経路を自動選択しない —
   機械や資産を替えたら黙って出力が変わる事態を禁じる。sha256 門（PNG / WAV / golden）は
   参照経路で凍結し、実装の正しさ・非破壊の機械検証器として温存する。
3. **沈黙縮退は「形・limit の適格判定」に限る**（linear の `k % 4`・attention の
   `D % 4`/`N % 4` と同じ流儀）。opt-in を指定したのに適格外で参照経路へ落ちる場合、縮退は
   許すが**診断のパイプラインキーに必ず出す**。未実装の組合せ（例: `linearCompute: "f16"`
   × i8 常駐）は縮退でなく fail loudly — 「指定が黙って無視される」と「形が適格でない」を
   混同しない。
4. **各席の検証門は 3 点セット MUST**: ①参照経路の既存 sha 門が無変更で緑（凍結の証明）
   ②opt-in 経路の A/B 帯門（参照経路との誤差帯 — 実測ドリフト × マージンで導出し、導出表を
   テスト定数の docstring に残す既存流儀。latent 門 w8 席 1e-2 が前例）③census 門（席を
   指定した時に変種が実際に効いたことの検査 — 全 dispatch が縮退していたら赤。
   `assets_fusion_counts_test` と同じ「性能だけ黙って落ちる」対策）。
5. **品質管理の分担**（ユーザー方針 2026-08-13 の ADR 化）: 実装の正しさ・非破壊は
   ビット同一で機械検証し、ビット同一が保証されない opt-in 席の品質は**人間レビュー
   （聴感・視認）で管理する**。数値指標（LSD 等）は助言的 — 数値が大幅に悪化していても
   生成結果が崩壊していないことは多い（w8a8 既定化の「LSD 5.64 より聴感が正」が前例 —
   ADR 0050）。
6. **既定への昇格は品質裁定とセット**: opt-in 席を既定へ昇格させる時だけ人間品質裁定を
   行い、sha 門の参照 digest 更新と既定値変更を**同一コミット**で行う（ばらすと門と既定が
   食い違う窓ができる）。前例 = w8a8 既定化（ADR 0050・ユーザー聴感裁定）。
7. **quant 席（格納形）との交差**: 実行席は格納形と直交し、組の適否は各席が自分で宣言する
   （`linearCompute: "i8a8"` × i8 常駐が実例・不能な組は決定 3 の fail loudly）。品質裁定が
   要るのは実行席・格納席それぞれの導入時と昇格時で、組ごとの全数裁定はしない（組が積に
   なって爆発する — 帯門と census が組の健全性を機械側で持つ）。
8. **最初の入居者 = K-5 / K-2**（席の命名と変種設計は各実装の設計提案で確定 — 本 ADR は
   契約のみ）。K-5 = Irodori DiT の masked online attention（目的は**ポータビリティ** —
   S=750 の中間 `scores[1,20,750,2269]` = 136,140,000 B ≈ 129.8MiB が WebGPU core 既定の
   maxStorageBufferBindingSize 128MiB を超える。一次出典 =
   [dit-export-recon](../research/2026-08-11-dit-export-recon.md) と
   [limitations](../limitations.md) の DiT 項。online 化は加算順が
   変わるため opt-in 席。[ADR 0044](0044-runtime-attention-mask.md) の safe_softmax 意味論の
   online 形への再導出が設計課題）。K-2 = VAE conv2d の Winograd / i8a8（kill 基準 =
   参照比 1.3 倍未満 — perf-ledger K-2 行）。

   **Follow-up（2026-08-15）**: 決定 8 の見込みどおりには入居していない。K-5 の目的
   （ポータビリティ）は前段 **K-5a = 行ブロック実行（[ADR 0060](0060-row-block-attention.md)）が
   ビット同一・既定経路で達成**したため本 ADR の席を使わずに閉じ、後段 K-5b（online / 鎖融合）は
   **スパイク実測で棄却**（[perf-ledger](../perf-ledger.md) K-5b 行 — online は LLM 波へ保留）。
   K-2 は整理整頓波の後。**本 ADR の契約自体は次の入居者を待つ状態で有効**。

## 追記（2026-08-29）— 決定 2 の例外条項: ビット同一主張の自動選択は機械検証つきに限る

`i8a8Dot`（`dot4I8Packed` / エミュの変種選択）は「両変種はビット同一 = 数値を変えない速度
選択」を根拠に決定 2 の対象外とし、アダプタの言語機能列挙から自動選択していた。この前提が
**融合 attention だけ Apple M2 で反証**された（[known-issues](../known-issues.md) — linear は
通る・同じ `idot` 生成を共有するため、故障は変種単体ではなくカーネル文脈〈幾何・タイル充填〉
との相互作用）。裁定:

1. **knob を族で分離**: `linearI8a8Dot` / `attentionI8a8Dot`（`SessionState`）。linear は
   従来どおり列挙で決める（反証が無い）。テスト専用 `I8A8_DOT` は両族を強制する意味を維持。
2. **attention は実走カナリアで決める**（`src/gpu/attention-dp4a-canary.ts`）: production
   幾何で 1 タイル全域 + K 2 タイルの固定入力を、生成されうる全 6 変種ぶん 1 submit で撃ち、
   TS 参照（`src/reference/i8a8.ts`）の既知解と atol=0 で突合する。dp4a 一致 → dp4a /
   dp4a のみ不一致 → attention だけ emu（縮退は決定 3 どおりパイプラインキー `:dp4aEmu` に
   出る）/ 両不一致 → `GpuFeatureError`（変種選択の問題ではなく整数 attention 自体が信用
   できない）。判定は device 単位 1 回（Promise メモ化）・`attentionCompute: "a8"` を要求した
   最初の Session 構築時のみ走る。
3. **一般則**: 「ビット同一だから決定 2 の対象外」を主張する自動選択は、その同一性を**当該
   device で機械検証できている**（実走カナリア等）場合に限る。列挙・仕様上の主張だけでは
   足りない — 本件が反例。

制約: カナリアが M2 の実故障を検出できるかは**未立証**（故障条件が不明のため。極小形を避け
production 幾何にしたが、B·H=1 の固定形が再現条件を外す可能性は残る — 実機確認の手順と
読み方は known-issues の Metal 節）。原因確定（backlog later）とは独立の封じ込め。

## 将来課題（記録のみ・着手未定）

- **一括フラグ**（「opt-in の高速化を全て使う」— ユーザー要望 2026-08-13）: 個別席が
  数個溜まってから設計する。宿題は ①明示の個別指定との合成規則（明示が勝つ）②意味が
  バージョン間で動く問題（新しい席が入るたび「全て」の中身が変わる — 再現性を求める
  利用者には個別指定を案内する等の線引き）③帯門の組合せの持ち方。
