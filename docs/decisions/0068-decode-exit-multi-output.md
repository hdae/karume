# 0068: decode 出口 — ノードレベル multi-output と argmax / static-k topk

- Status: accepted（2026-08-17 — 委任チェック方式・大域裁定なし。Codex レビュー
  第 3〜5 巡を反映〈出力 0 本の定義域・ir-v1 改訂要件・topk 受理領域〉し第 6 巡で go）
- 関連: ADR [0066](0066-generation-context-state-slots.md)（DDS 席の前提 = 複数出力）/
  [0059](0059-op-vocabulary-entry-doors.md)（op 追加の入場門）/
  [0058](0058-numerics-opt-in-contract.md)（検証門 3 点セットの流儀）
- 根拠:
  [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  §2（以下「調査 §n」）

## Context

decode 1 step の出口で「全語彙 logits をホストへ readback」する形は参照実装で既に少数派
（llama.cpp は readback 自体を発行しない・web-llm は int32 1 個・調査 §2）。karume で
greedy / top-k を GPU 側に置くには **ノードレベル多出力**が要る — グラフ出力レベルの
multi-output は既に動くが、ノードは 6 面で単一出力が前提（contracts / plan / recipe /
recipe-builder / executor / exporter — 調査 §2・第 2 巡で確定）。この 6 面の解禁は
Kokoro-82M（LSTM h_n）と DDS 席（payload + extent — ADR 0066 決定 3）の前提でもある。

## Decision

### 1. ノードレベル多出力の解禁（IR 仕様は無改訂・6 面の実装改訂）

IR スキーマは `outs` 長さ 1 以上を既に許可しており**仕様改訂は不要**。改訂するのは実装 6 面:

- 契約テーブル（runtime `ops/contracts.ts` / exporter `ops.py`）: **出力 slot 別の
  dtype / shape 写像**欄を追加する（現行の「スロット 0 → 出力・恒等」の一般化）。
  `assertNodeContract` の `outs.length !== 1` 門は「契約が宣言する出力数と一致」へ。
- **定義域は 0 本を含む**（第 3 巡で追加）: 値を定義しない effect op（ADR 0067 の
  `state_append`）は**出力 0 本を契約で宣言**する。IR パーサの「outs 空は拒否」
  （format/ir.ts:322-324 —「値を定義しないノードは静的 DAG に置けない」）は「契約が
  effect を宣言する op に限り 0 本を許す」へ改訂し、実行順はデータ辺ではなく
  nodes 配列順契約（ADR 0067 決定 5b）が持つ。recipe の出力確保・retain・解放簿記は
  0 本 / 多本の両方向へ一般化する（単一出力の生成物バイト不変は維持）。
  **精密化（第 4 巡）**: 「IR 仕様は無改訂」が言えるのは**複数 outs**についてだけで、
  **0 本の解禁は ir-v1.md の改訂を要する**（本文が outs ≥ 1 を明記している — 実装波で
  本 ADR とセットの本書改訂として行う。未リリース改訂手順どおり version 1 のまま）。
- plan / recipe / recipe-builder / executor: `outs[0]` 前提（plan.ts:81,431・
  StepRecipe の単一 outputName / output / uses — recipe.ts:84-100）を出力列へ一般化。
  **単一出力ノードのレシピ表現・生成物・診断はバイト不変 MUST**（表現の一般化で既存
  経路のスナップショットを動かさない）。
- 出力ごとに dtype が異なる形（topk = 値 f32 + index i32）を契約層で表せること。

### 2. `argmax`（greedy の MVP — 単一出力・先行導入）

- 契約: 最終次元の argmax・入力 f32・出力 **i32**・rank 保存（`keepdim` 相当は欄なし =
  最終次元を 1 に潰す固定形）。
- MUST: **タイブレークは最小 index**（torch 準拠）。llama.cpp は GPU 側 = 最大 index /
  CPU sampler = 最小 index で**同一リポ内で食い違っており**（調査 §2）、明文化しないと
  greedy の再現性が実装差で割れる。
- MUST: 行 max の初期値は **−inf**（有限 sentinel 禁止 — index 追跡と組で全 −inf 行も
  「最小 index = 0」の決定的挙動になる。ADR 0067 決定 6 と同じ理由）。
- 実装形は既存 row-reduce と同型（grid-stride ローカル max+index → 共有メモリツリー簡約。
  index を運ぶ点だけが新しい）。

### 3. `topk`（multi-output の最初の入居者）

- 契約: 最終次元の top-k・**k は attrs（宣言必須・計画時定数 = static-k）**・出力 2 本 =
  値 f32（**降順ソート済み** — torch 同値）+ index i32。
- MUST: 全語彙 argsort を経由しない（MLC の WebGPU 経路が高コスト側の実例 — 調査 §2）。
  実装は block-local top-k → merge の形（llama.cpp WebGPU の argsort+merge 骨格から
  k 幅に絞る）。scratch が要る場合は一時バッファとして recipe に載せる（出力バッファへの
  同居〈llama.cpp 流〉は採らない — 確保仕様が読めなくなる）。
- タイブレークは argmax と同じ「最小 index 優先」（同値要素の順序も torch と一致させる）。
- **受理領域（第 3 巡で追加）**: `1 ≤ k ≤ 最終次元`。k=0・k > 最終次元・記号 k は
  fail loudly。実装上限（workgroup storage 等の device limit から静的に決まる k の上限）を
  超える指定も **fail loudly**（縮退しない — ADR 0058 決定 3 の「未実装の組」と同じ扱い。
  上限値は実装が診断つきで報告する）。runtime / exporter の受理集合は契約テーブル 1 本から
  導出し、6 面で食い違わせない。

### 4. decode グラフの出口形

- lm_head は**最終位置のみ**に通す（vLLM 型 — 調査 §2）。karume ではこれは op ではなく
  **エクスポータの decode グラフ台本**が自然に持つ（decode は queryLength=1 なので追加
  機構は不要。prefill チャンクの途中 logits は出力しない）。
- readback は sampled token（argmax 出力 i32）または topk の 2 本のみを既定にする。
  全語彙 logits の readback は「グラフ出力に logits を宣言した場合」の opt-in として
  残す（logprobs 用途 — 欄を消さない）。
- **sampling / RNG はホスト維持**（op-vocabulary の裁定を再確認 — GPU 側は argmax / topk
  のみ。温度・top-p・乱数は topk 読み出し後のホスト演算）。

## 検討した代替案

- GPU 側 full sampling（gumbel-max / top-p renormalize — vLLM・web-llm 型）: readback を
  int32 1 個まで削れるが、乱数の再現契約（seed 管理）と検証面が一気に広がる。topk k 本の
  readback（k ≤ 64 で 512B 級）はホスト sampling で十分小さい。却下（将来の性能実測で再訪）。
- 多出力を「複数の単一出力ノードへ分解」（topk_values / topk_indices の 2 op）: 契約は
  単純だが同じ縮約を 2 回走らせるか、暗黙の共有 state を op 間に持つかの二択になり、
  どちらも劣る。LSTM h_n / DDS extent には分解の余地自体が無い。却下。

## Consequences

- Kokoro-82M（LSTM = h_n 出力）と DDS 席（ADR 0066）の前提が開通する。
- 契約テーブルの出力写像欄は全 op に入る（既存 op は「1 出力・恒等」の明示化 —
  挙動不変・表の形だけ変わる）。exporter の `len(node.outs) != 1` 門は契約参照へ差し替え。
- 受入条件（実装波のゲート）: ①既存単一出力経路の生成物・診断バイト不変（スナップショット
  無風）②argmax / topk の CPU 参照オラクル一致 + タイブレーク直接門（同値要素・全 −inf 行）
  ③故障注入（index ずれ・merge 境界・k 端数）④multi-output の寿命検証（片方だけ消費される
  グラフで uses / release が正しいこと）。

## 追記（2026-08-17・実装波スカウトの補正）

決定 1 の「実装 6 面」は現物では **8 面**: 列挙した 6 面に加えて **fusion**
（`fusion.ts` — FusedStep の単一出力前提と適格条件 3 本）と**契約適合表 fixture**
（`packages/runtime/tests/fixtures/op-contracts.json` — TS / Python 両実装が読む唯一の
正本 schema）が独立の改訂面として立つ。executor は逆にほぼ無風（グラフ出力レベルの
multi-output は実装済み）。exporter 側は「多出力 aten を通す道が現状ゼロ」で、converter の
タプル meta + `operator.getitem` スロット結線が新機構になる（argmax は単一出力なので
この機構を要しない — 段階分割の自然な切れ目）。
