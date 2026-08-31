# 0044: 実行時 attention マスク — bool 入力 + safe_softmax（ガード証明の構成的置換）

- Status: accepted
- Date: 2026-08-11
- 関連: ADR [0016](0016-anima-chain-export.md)（safe-softmax ガード除去 = 実値証明）/
  [0023](0023-fused-attention.md)（融合 attention の mask 契約 — 静的 `[1,1,M,N]`）/
  [0009](0009-dtype-i32-bool.md)（bool = u32 0/1・IO 可）/
  [0043](0043-op-addition-layers.md)（層判定 — safe_softmax は第 2 層）

## Context

- Irodori-TTS v4 の CFG は uncond を「条件 state = 0 + **マスク全 0**」で作る（ユーザー裁定
  2026-08-11: 実行時 bool マスク入力を許す = 案 a。recon =
  [research/2026-08-11-irodori-source-recon.md](../research/2026-08-11-irodori-source-recon.md)）。
  現行設計はこれと正面衝突する: SDPA 分解経路の safe-softmax ガード（「全要素 -inf の行は
  0 を返す」）を除去するために `_assert_guard_inactive`（`karume/normalize.py`）がマスクの
  **実値評価**を要求し、`_eval_static` は実行時入力の placeholder に触れた時点で fail loudly
  する。マスクが実行時値なら証明が原理的に立たない。
- Irodori の実測形（`irodori_tts/model.py` HEAD `8224daf`）: 全 attention のマスクは
  **key-only `[B,1,1,K]`**（`attn_mask = cat(context_masks)[:, None, None, :]` —
  model.py:403-404、および `key_mask[:, None, None, :]` — model.py:189-191）。joint attention
  の cat 先頭は `self_mask`（省略時 `torch.ones(bool)` — model.py:357-358）なので、CFG で
  条件側マスクが全 0 でも**行が全 -inf になることは構造上ない**。
- EmbeddingGemma の「実行時 attention_mask 非対応」（limitations — バッチ内パディング不可）
  も同じガード証明が根: 機構としては本 ADR で解ける（配線は EG 波で別途）。
- torch の SDPA 分解は bool マスクを `where(mask, 0, -inf)` で加算型へ変換してから
  `add → softmax → ガード` を発行する。karume 語彙の現状: `where` は
  **スロット契約 `[BOOL, F32, F32]` で既存**（ops.ts）、`scalar_tensor` は FOLDABLE
  （0 / -inf は f32 **バイナリ** initializer に畳まれる — IR の JSON 層に非有限値は
  現れない。EG の帯マスク定数と同じ前例）、`cat` は f32 専業（bool の cat は不可）。

## Decision

### 1. マスクの IR 境界 = bool 入力（連結済み・graph 内で加算型へ変換）

- グラフ入力は **bool（u32 0/1・ADR 0009）** のマスクをそのまま受ける。torch が分解時に
  発行する `where(mask, 0, −inf) → add(scores, ...)` を**そのまま IR に写す**（where は
  既存語彙・0/−inf は f32 initializer・同一マスクの変換は CSE で 1 本に畳まれる）。
- **bool の cat は語彙に足さない**: Irodori の「セグメント bool マスクの cat」はホストが
  連結済みの 1 本（`[1,1,1,K]`）を渡す形に export 台本で正規化する（セグメント長は
  呼び出し側が各セグメントの context を組む時点で既知 — 知識の複製はほぼゼロ）。
- 融合 attention（ADR 0023）の mask 契約は**変えない**（静的 `[1,1,M,N]` のみ）。実行時
  マスク付き SDPA は分解経路で実行する（Irodori recon の判断どおり）。融合側の実行時
  マスク対応は性能要求が実測されたら別 ADR。

### 2. ガードの扱い = 実値証明 → 不能なら safe_softmax（第 2 層 op）へ書き換え

- `_drop_safe_softmax_guard` を 2 段にする:
  1. 従来どおり `_assert_guard_inactive`（実値証明）を試み、成立すればガードを**除去**
     （既存資産の出力バイト列は不変 — EG / Anima の経路はここで閉じたまま）。
  2. 証明不能（実行時マスク等）なら fail loudly せず、ガード部分木
     `where(¬any(¬eq(src,−inf)), 0, softmax(src))` を **`safe_softmax(src)` 1 ノードへ
     書き換える**。matcher は既存 `_guard_parts` を流用。
- `safe_softmax` の契約: softmax と同一 + 「**行 max が −inf の行は全 0 を書く**」。f32。
  有限要素を持つ行の計算経路は softmax と同一（ビット同一はスナップショット + parity で
  固定）。全要素 −inf ⇔ 行 max = −inf は「スコアは有限」（非有限入力は契約外 —
  limitations）の下で同値なので、torch ガードの意味論と厳密一致する。
- ガード部分木の op 化（eq(−inf) 等を語彙に足す案）は ADR 0016 の却下を踏襲
  （`eq` のスカラ −inf が IR JSON 層の非有限値拒否と衝突する）。

### 3. 入力契約

- 加算マスク値は where 変換由来の {0, −inf} が正規。行が全 −inf（= bool 全 0 の行）は
  **safe_softmax 経路でのみ正規**（出力 0 行 — torch と同値）。plain softmax / 融合
  attention に全 -inf 行を渡すのは契約違反で NaN 汚染（gather の範囲外と同じ扱い —
  検査は入れない）。

## 検討した代替案

- **ホスト構築の加算 f32 マスク入力**（bool を IR に持ち込まない）: グラフは add 1 本に
  なるが、0/−inf 化のホスト実装 + Python 側とのパリティテスト（SBV2 relattn 表と同じ
  機構一式）が要り、IR 入力の意味も dtype から読めなくなる。where 1 dispatch
  （K 要素・CSE 後 1 本）の節約に見合わない。却下。
- **構成的不活性証明器**（「cat の先頭セグメントが定数全 1」等を graph パターンで証明して
  従来どおり除去）: 証明器が transformers / モデル毎のマスク構築形に脆く（融合 matcher の
  「黙って外れる」と同型の脆さを正しさ側に持ち込む）、空行が正規に出るモデル（バッチ pad・
  EG の将来形）に閉じない。safe_softmax なら意味論で閉じ、証明は不要。却下。

## 追記（2026-08-11・実装時の明確化）

決定 2 の「証明不能」には、証明が立って**「不活性でない」ことが確定した**形（全 −inf 行が
実在する静的マスク）も含める。ガードは除去ではなく実装（safe_softmax）へ置き換わるので、
どちらの系でも NaN が下流へ流れる経路は生じず torch と同値のまま。ADR 0016 下で
fail loudly だったこの形は正規の受理形になった（実測 =
`test_an_all_masked_row_rewrites_the_guard_to_safe_softmax` ほか 2 系）。

## Consequences

- Irodori DiT export（第 2 波）が開通: 語彙追加は `safe_softmax` 1 op（第 2 層 — 契約
  2 実装 + WGSL + CPU 参照 + 適合表 + golden + 分解経路とのビット同一 parity が 1 セット）。
- 既存資産は不変: 証明成功経路の出力・既存 IR ファイルとも影響ゼロ。safe_softmax は
  「証明不能」の新経路にだけ現れる。
- EG の実行時 attention_mask（limitations）解消の道筋が確定: pad マスクを bool 入力で
  受け、帯定数と**加算合成**（0/−inf の加算は順序不変で丸め誤差なし）。SDPA を保存
  （融合）のままにするか分解へ落とすかは EG 波で裁定（融合契約を広げるか、その変種だけ
  分解で受けるか）。
- Irodori 第 2 波の残課題（本 ADR の範囲外として明示）: `_safe_attention_mask`
  （model.py:429 — 空行の先頭を強制 True にするホスト側ロジック）は B=1 + 「参照ありなら
  マスク非空」の入力契約で第 0 層 patch、duration の masked_mean（caption 全 0 時の 0/0）は
  export 台本側で処置。

## 追記（2026-08-18・対照 — states 形は空行が正規）

決定 3 の「融合 attention へ全 −inf 行を与えるのは契約違反」は **states 形（ADR
[0067](0067-autoregressive-attention-vocabulary.md) 決定 4 — `states` 欄つき attention）には
適用されない**。states 形では padding 行が窓から落ちる空行が**正規に**出るため、空行 →
出力厳密 0 を構造的に保証する（決定 6 — 有限 sentinel での代用は MUST NOT。保証の実体は
「空行 ⊂ pad 行」に対する pad 行の 0 書き — ADR 0066 追記 8。行統計の「identity −inf +
空行 `(0,0)`」ガードは防御専用）。2 つの契約は同一 op 名の別形として並立し、
欄の有無が形を判別する。

## 追記（2026-08-31 — OP 数値レビュー W-2 / W-3 の裁定 a）

決定 3 の「plain softmax / 融合 attention に全 -inf 行を渡すのは契約違反で NaN 汚染」の
うち、**融合 attention 側は撤回する**: attention_stats v2 が safe_softmax のガード 3 点
（identity −inf・空行判定・空行 stats (0,0)）を移植し、全マスク行は**出力 0 の正規入力**に
なった（CPU 参照 referenceAttention も同じ空行分岐を持つ）。有限要素を 1 つでも持つ行の
ビット列は不変なので、分解経路（plain softmax）との parity は従来どおり成立する。
plain softmax への全 -inf 行は引き続き契約違反（不変）。

併せて softmax / safe_softmax / attention_stats / attention_state_stats の行 max を
`nan_max`（ADR 0020 のビット列判定）へ統一した。素の `max` は WGSL 仕様レベルで NaN を
落とすため、**全要素 NaN の行が safe 系の空行判定に化けて厳密 0 になり NaN が黙って消えて
いた**（部分 NaN 行は総和経由で従来から伝播 — 穴は全 NaN 行だけ）。v2 では全 NaN 行が
NaN のまま出力へ流れる（fail loudly と limitations の伝播規約に整合）。非 NaN 入力の
ビット列は全変種で不変（golden / WAV / PNG 門はそのまま）。
