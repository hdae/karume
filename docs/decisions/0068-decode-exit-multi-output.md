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

## 追記 2（2026-08-17・実装波 B での実測訂正 3 点）

1. **決定 3 の括弧書き「同値要素の順序も torch と一致させる」は実測で不成立**
   （torch 2.13.0+cpu）: torch の `topk` は tie の順序を保証せず、`topk([5,5,5,5],1)` は
   index 2・`argmax` は 0 で **torch 自身が同一リポ内で食い違う**（多値 tie では降順
   index も観測）。採った規律 — **値の列は torch と数値同値**（降順・多重度一致。比較
   同値類がビット単位でも同一なら bit 一致し、実測 200×4 ケースは全て該当。**±0.0 の符号
   ビットと NaN payload は選択添字依存**で、karume は最小 index の要素のビットを書く —
   torch の非規範な tie 選択とはビットが割れうる。第 2 巡レビューの反例で精密化）・
   **添字の列は karume が最小 index に規定**（決定 2 の argmax と同族・決定的。
   k=1 が argmax と一致することを門が突き合わせる）。
2. **実装上限の具体形**: scratch を workgroup storage に閉じる実装（temps 不使用 —
   決定 3 の条件付き記述は不発動）を採った結果、上限は `8·W·(k+1) ≤
   maxComputeWorkgroupStorageSize`（W=32）で **WebGPU 既定 16384B では k ≤ 63**。
   超過は上限値・必要バイト数つきで fail loudly（縮退しない）。device 依存の段差は
   limitations に起票。
3. **exporter 多出力の停止点は `operator.getitem`**（実測）: aten handler を足しても
   torch.export がタプル返しへ挟む getitem で変換が止まる — 「タプル meta + getitem
   スロット結線が新機構」（追記 1）の裏付け。topk の aten handler は sampling 実需まで
   先送り（2026-08-17 裁定）で、契約表・kernel・検証門のみ先行実装。

## 追記（2026-08-17・実装波スカウトの補正）

決定 1 の「実装 6 面」は現物では **8 面**: 列挙した 6 面に加えて **fusion**
（`fusion.ts` — FusedStep の単一出力前提と適格条件 3 本）と**契約適合表 fixture**
（`packages/runtime/tests/fixtures/op-contracts.json` — TS / Python 両実装が読む唯一の
正本 schema）が独立の改訂面として立つ。executor は逆にほぼ無風（グラフ出力レベルの
multi-output は実装済み）。exporter 側は「多出力 aten を通す道が現状ゼロ」で、converter の
タプル meta + `operator.getitem` スロット結線が新機構になる（argmax は単一出力なので
この機構を要しない — 段階分割の自然な切れ目）。

## 追記 3（2026-08-18・波 E — 決定 4 の既定出口は未検収・opt-in 形で先送り）

波 E の decode 台本（minicpm5 `export_decode.py`）は決定 4 の既定（lm_head 最終位置のみ・
readback は token のみ）ではなく、**全 M 行に lm_head を通し logits + token を必須出力に
宣言する opt-in 形**で検収した。機序: 「最終位置のみ」は decode（M=1）では自明だが、prefill
チャンクの最終**有効**行 = `queryLength − 1` は実行時スカラで、現行 IR 語彙にはグラフから
これを静的に切る手段が無い。既定形の成立には last_row 添字を i32 グラフ入力で受けて
gather → lm_head へ通す新配線（+ models 側の入力供給・token-only 検収門）が要る —
独立の設計判断として切る。

裁定（2026-08-18）: **token-only 既定形の追加は波 F/H で行う**（backlog 4 番に起票）。
現系列は診断線（prefill logits tolerance 門の対象）として併存させる。それまで decode の
実効 readback（prefill チャンクあたり ~16.7MB の logits）と全行 lm_head 計算は過大のまま —
検収の正しさには影響せず、この形の上で decode 性能を主張しない。

## 追記 4（2026-08-19・波 H — 決定 4 の既定形〈token-only 出口〉を実装）

追記 3 で切り出した token-only 既定形を Gemma 4 E2B で実装・検収した
（`tools/export-recipes/gemma4/export_token.py` — 系列 `gemma4-e2b-decode-token/`）。

- **配線は追記 3 の想定どおり**: `last_row[1]` i32 グラフ入力（最終有効行 = `queryLength − 1`・
  models `generateGreedy` の `lastRow` 指定が供給）→ 最終 norm 後 hidden `[1,M,H]` から
  `F.embedding(last_row, hidden[0])` で 1 行選択 → その行だけ lm_head + softcap + argmax →
  出力は `token[1,1,1]` **1 本**。**新規 op ゼロ**（行選択は既存 `embedding` — 添字が実行時値
  でも最終次元固定の行 gather。`logits_to_keep` にテンソルを渡す上流形は advanced indexing =
  語彙外なので使えない）。IR / ランタイム / エクスポータ core は無変更。
- 出口は **argmax 直結 MUST**（形合わせの unsqueeze は lm_head の前に置く — 後ろに置くと
  token 出力の供給元が reshape になり形検査が割れる）。
- **検収は系列間交差 parity**（`e2e_gemma4_token_exit_test.ts`）: logits opt-in 系列の
  greedy golden（torch full re-forward）と token-only 系列の生成列が 3 ケース × K=16 で
  厳密一致。期待値の再計算は払わず、独立性は落ちない（出所は full re-forward のまま）。
- 実効: decode の readback は token 4B のみ・lm_head は 1 行（prefill chunk あたり
  `(M−1)×262144×1536` MAC と `[M,V]` logits バッファが消える）。logits opt-in 形は診断線
  （prefill logits tolerance 門の対象）として併存（追記 3 の裁定を維持）。
- 送り: MiniCPM5 系列への同形展開（backlog 起票 — 機構は models `lastRow` として共通化済み）。

## 追記 5（2026-08-19・全体レビュー CX-2.3 — token-only 系列の出所束縛）

追記 4 の検収（系列間交差 parity）は「両系列が同じチェックポイントから出た」という前提に
立つが、その前提だけが機械可読でなかった — 資産ディレクトリの存在確認しか無く、片方だけ
古い組み合わせでも門が緑になれた。消化（コミット `353baf0`）:

- **token-only の export が `reference.json` を書く**（書き手の正本 =
  `tools/export-recipes/gemma4/provenance.py`）: 元チェックポイントの指紋
  （model/config/tokenizer の sha256 + bytes）と、流用する `greedy.<case>.safetensors`
  1 本ずつの digest。golden の `prompt` は export 時に `torch.equal` で今回のケースと突合
  （`expected` は読まない — 期待列の突合は実 GPU 門の仕事のまま）。
- **検収門に③（出所の束縛）を追加**（`e2e_gemma4_token_exit_test.ts`）: 記録の schema・
  系列名・golden 集合の過不足・実バイトの sha256 照合。記録なし / 不一致は SKIP でなく
  **FAIL**（logits 系列を採り直せば digest が動き、token-only の再 export が強制される）。
- **据える単位を token-only でも「系列ディレクトリ丸ごと」へ統一** — 容器と記録が同じ
  据え替えで動くので「新しい容器 + 古い記録」が構造的に作れない。logits 系列の再 export は
  不要（既存 golden のバイトを読むだけ）。
- 出力レイアウト: `gemma4-e2b-decode-token/` = `model.safetensors` + `reference.json`。
- 送り: logits opt-in 系列側の同形記録（両系列が同じ checkpoint を名乗ることの機械照合）は
  次にその系列を採り直すときに同時導入する（greedy 再採取が高価なため — レビュー隣接記録）。

## 追記 6（2026-08-31・生成 API 波 — 製品グラフの既定出口は「最終行 logits」）

決定 4 は readback の既定を「sampled token または topk の 2 本」とし、全語彙 logits を
**グラフ出力に logits を宣言した場合の opt-in として残す（欄を消さない）**と書いた。生成 API 波の
設計裁定（2026-08-31）で、**gemma4 の製品グラフはこの opt-in 側を既定にする** — ただし全 M 行では
なく **`last_row` で選んだ最終行だけ**の `logits[1,1,V]` を出す形である（正本 = ADR
[0083](0083-generation-api-surface.md) 決定 6）。決定 4 の欄をそのまま使うので**本 ADR の決定は
撤回しない**（既定の選び方が「token-only」から「最終行 logits」へ移るだけで、topk 出口と GPU 側
sampling の扱いは不変）。

- **配線は追記 4 の実装から argmax を外すだけ**（`export_token.py` の `TokenOnlyChunkWrapper` —
  最終 norm 後 hidden から `last_row` 行を `embedding` で選び、その 1 行に lm_head + softcap を
  通す）。新規 op ゼロ・IR / ランタイム / エクスポータ core は無変更のまま。
- **topk 出口は実需が立たない**ので保留を続ける（`operator.getitem` 配線も同様 — 追記 3）。
  理由は**追記 2 の実装上限 k ≤ 63 が、gemma-4-E2B-it の `generation_config.json` の推奨既定
  `top_k: 64` に 1 だけ足りない**こと（実資産の実測）。本 ADR の「検討した代替案」欄が却下の
  比較対象を「topk k 本（k ≤ 64 で 512B 級）」と書いていたのに実装が 63 に着地したので、その 1 の
  差がそのままモデル既定と噛み合わない。加えて repetition penalty / logit bias / full-vocab
  nucleus は全語彙を要求する。
- **決定 4 の「sampling / RNG はホスト維持」は不変**（GPU 側 full sampling は却下のまま — 採るなら
  再裁定）。ホスト sampler の置き場と契約は ADR 0083 決定 7。
- **読み戻しの実効**: decode は 4B → 1 MiB へ増えるが、同じ submit に相乗りするのでフェンスは
  増えず、262,144 要素の JS 走査込みで 0.3〜0.6ms 級の見積り（decode 32.5ms/token = ADR 0082 に
  対して数%）。**prefill はむしろ減る**（logits opt-in 系列の `[1,M,V]` は chunk 32 で 32 MiB →
  最終行だけなら 1 MiB）。追記 3 が「この形の上で decode 性能を主張しない」と書いた過大な
  readback は、最終行に絞ることで再来しない。
- 既存 2 系列（logits opt-in / token-only）は**検収 fixture として併存**させる（追記 3 の裁定を
  維持）。製品グラフ 1 系列への集約は生成 API 波の段 1b で、PLE 外出し（ADR
  [0085](0085-ple-host-gather.md)）と同じ再 export に載せる。

## 追記 7（2026-09-08・MTP 段 1 — 製品グラフの出口は「選んだ R 行の logits + hidden」）

追記 6 の「最終行 logits `[1,1,V]`」を一般化した（ADR 0096 決定 5）。`last_row` は `[R]`（第 2 記号
R・要素数がその run で選ぶ行数を束縛する唯一の源）、グラフ出力は **出力 0 = logits `[1,R,V]`・
出力 1 = 最終 norm 後の hidden `[1,R,H]`** の 2 本（順序が契約 — ランタイムはスロット番号で読む）。
通常の prefill / decode は R=1 で、値も token 列も従来の 1 行出口とビット同一。投機 verify は
R = k+1 行を 1 run で採点し、hidden の受理行を drafter の入力にする。R を chunk 行数 M と共用しない
（prefill が `[1,768,V]` を readback してしまう）。行選択が lm_head より前に居ることの構造検査
（H-01）は行数を引数で受ける形（`assert_row_selected_lm_head`）に一般化した。GPU 側 argmax の禁止
（追記 6・ADR 0083 決定 6）は不変 — 縮約は実測してから。

## 追記 8（2026-09-09・MTP 段 4-B ③ — 長い行の argmax は 2 dispatch）

決定 2 の argmax は「1 行 = 1 workgroup」で、行が語彙長（gemma4 262,144）になると 256 スレッドが
1,024 要素ずつを逐次で畳む遅延が律速だった（drafter の 3 段で 1.5〜2.1 ms/cycle — research
2026-09-09 §4）。行長 ≥ 16,384（4,096 要素の区間 4 本以上）は **partial（区間ごとの最大元を一時
バッファへ）→ merge（行ごとに区間の結果を畳む）** の 2 dispatch にする。結果は 1 dispatch 形と
**ビット同一**（(値 降順, index 昇順) の辞書式順序の最大元は結合順に依らない・区間の identity は
同じ −inf / 番兵 `dim`・区間は空にならない）。経路の選択は行長の純関数（`argmaxSplitGroups`）で
実行相と見積り（一時 `[rows, groups] × 8 B`）が同じ関数を通る。短い行はキー・WGSL とも不変。
タイブレーク / NaN / 全 −inf 行の規定（追記 2）は 2 相形の門（`gpu_ops_test.ts`）でも同じ
リテラルで固定した。GPU 側 argmax の禁止（追記 6・7）は target の出口の話で不変 — 2 相化は
drafter が既に持つ argmax 出口の速度だけを変える。

## 追記 9（2026-09-11）: topk k=1 の長い行も 2 dispatch へ分割する

- 出力転送を減らす実験で、既存topkのk=1経路が大語彙では費用を増やすと判明した。
  argmaxの部分最大・mergeを共有し、k=1かつ行数が正で行長16,384以上を2 dispatchへ分割する。
  [単体とモデル全体の実測](../research/2026-09-10-codex-mtp-optimization.md#topk-k1-の分割と小出力の実験2026-09-11)を根拠に採用する。
- 比較順序は従来と同じ（NaN優先・値降順・最小index）。値の出力は選ばれた元入力のu32を写し、
  ±0とNaN payloadを保つ。中間のf32最大値を値出力へ変換する形にはしない。
  argmaxの生成物はバイト不変。k>1と短い行のtopkも変更しない。
- 選択関数`topkOneSplitGroups`を実行と見積りの双方で使う。65,535区間を超える行は
  従来のgrid-stride 1 dispatchへ残し、対応入力を狭めない。
  partialの一時領域は`rows × groups × 8 B`。mergeまで元入力・partialを保持し、
  値と添字は独立した出力領域へ書く。実行と見積りの寿命を一致させる。
- 公開API・IR・保存形式は変更しない。Sessionが既存topkノードを実行する際の内部最適化。
  Gemmaの製品資産とCLIの出口は引き続きlogits+hiddenであり、これだけで転送量は減らない。
  小出力化の統合は会話状態のリース・バッチ完了・sampler能力を含む別の設計単位。
- 境界、NaN payload、同点、全−inf、符号付きゼロ、最終index、先行ノードの中間出力、
  backing再利用、メモリ見積りを検証する。GPU単体の倍率を生成全体の倍率と混同しない。
  M2での速度は未検収。
