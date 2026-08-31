# 0083: 生成 API 面 — `GenerationProgram` / `GenerationSequence`・token イベント・sampling ホスト

- Status: accepted（2026-08-31 — 設計ドラフトの裁定 10 点をすべて★推奨案で確定〈ユーザー裁定〉。
  実装は未着手 = backlog now の段 0〜3）
- Date: 2026-08-31
- 対象: `packages/models/src/generation/`（新規 `program.ts` / `sequence.ts` / `sampler.ts` +
  `greedy.ts` の格下げ）と gemma4 の製品グラフ台本。IR 仕様・ランタイム・hub は無改変。
- 関連: ADR [0066](0066-generation-context-state-slots.md)（GenerationContext の寿命・決定 6 の
  二重簿記の禁止・決定 8 のスコープ外・追記 2 の `rewind` 拒否）/
  [0068](0068-decode-exit-multi-output.md)（decode 出口 — 決定 4 が opt-in で残した全語彙 logits
  の欄を本 ADR が製品既定へ引き上げる。0068 側は**追記 6** で受ける）/
  [0067](0067-autoregressive-attention-vocabulary.md)（state 参照つき attention）/
  [0008](0008-public-api.md)（薄い公開面）/ [0078](0078-anima-sampler-selection.md)（配布形が
  既定を宣言しリクエスト席で上書きする前例）/ [0079](0079-sbv2-two-layer-input.md)（呼び手の
  責務分割の前例）/ [0082](0082-linear-gemv-decode.md)（decode 壁 32.5ms/token の実測前提）/
  本波の兄弟 = [0084](0084-gemma-tokenizer-chat.md)（tokenizer・chat）/
  [0085](0085-ple-host-gather.md)（PLE 配布形）
- 根拠:
  [research/2026-08-31-generation-api-design-draft.md](../research/2026-08-31-generation-api-design-draft.md)
  （以下「ドラフト §n」— 候補比較・棄却理由・実資産の実測はそちらが正本）

## Context

消費者が触れる生成面は `generateGreedy` 1 本（`packages/models/src/generation/greedy.ts`）で、
LLM 実需（streaming チャット）に要る面が構造的に欠けている:

- **静的配線とリクエストが 1 つの型に同居する**（`GreedySpec` — `session` / `inputIds` /
  `positionIds` / `token` / `lastRow` / `chunkLength` / `maxPosition` / `bindings` が静的側、
  `prompt` / `maxNewTokens` がリクエスト側）。
- **停止条件が無い**（「EOS 停止は載せない」と本文が書いている）。**sampling はホストにも無い**
  （「温度も top-k も RNG も置かない」MUST — 固定 token id 列での検収が目的の関数だから）。
- **多ターンの面が無い**（毎回 context を作り `finally` で dispose する）。

一方で足場は揃った: 同一 context への並行 run は既にランタイムが `ExecutionError` で拒否し
（レビュー修正波 CG5-2）、`last_row` 添字による最終行選択の配線は実装済みで（ADR 0068 追記 4）、
decode の壁は 32.5ms/token（ADR 0082）まで下がっている。

本 ADR は「文字列 in → 文字列 out」を消費者へ渡すための **API 面と契約**を固定する。tokenizer /
detokenizer / chat は ADR 0084、PLE の配布形は ADR 0085 が持つ。

## Decision

### 1. 面は 2 層 — `GenerationProgram`（静的配線）と `GenerationSequence`（1 会話の寿命）

`GenerationProgram` は **setup 時に全結線を検証する不変オブジェクト**、`GenerationSequence` は
1 会話ぶんの寿命を持つ実体。型は仮（実装波で確定）だが契約は次の形:

```ts
type GenerationProgram = {
  readonly inputIds: string;
  readonly positionIds: string;
  readonly lastRow: string;
  readonly logits: string; // ← 最終行 logits 出口（決定 6）
  readonly chunkLength: number;
  readonly maxPosition: number;
  readonly vocabSize: number;
  readonly stopTokens: readonly number[]; // EOS 集合（決定 8）
  readonly bindings?: SymbolBindings;
};

type GenerationRequest = {
  readonly prompt: readonly number[]; // 多ターンでは「今ターンぶん」だけ
  readonly maxNewTokens: number;
  readonly sampler?: SamplerSpec; // 省略時は配布形の宣言（決定 7）
  readonly signal?: AbortSignal;
};
```

MUST: **program は可変状態を持たない**。そして **sequence の可変状態は `context` と
`pendingToken` の 2 つだけ**である。position / totalLength といった counter を sequence に
持たせず、run を組む直前に `context.pastLength` を読んで `position_ids` を作る（ADR 0066
決定 6 = **二重簿記の禁止**。論理長の進行は run の成功で context が進める）。`totalLength` が
要るなら `context.pastLength + (pendingToken ? 1 : 0)` をその場で導出して**保存しない**。

### 2. token 列は `AsyncIterable<GenerationEvent>` で出す（裁定 1）

```ts
type GenerationEvent =
  | { readonly kind: "token"; readonly id: number; readonly position: number }
  | { readonly kind: "prefill"; readonly chunk: number; readonly chunks: number };

interface GenerationSequence {
  generate(request: GenerationRequest): AsyncIterable<GenerationEvent> & {
    /** 停止理由（`eos` / `max-tokens` / `aborted`）— iterable を汲み切った後に読む。 */
    readonly done: Promise<GenerationStop>;
  };
  dispose(): Promise<void>;
}
```

家風は anima 流の `onEvent` コールバック（`packages/models/src/anima/pipeline.ts`）だが、その
成立理由がここでは反転する: anima は「**返り値が画像 1 枚で、途中経過が副次**」（event union は
`stage` / `denoise-step` / `vae-tile` = 全部進捗）なのに対し、LLM は**列そのものが返り値**である。
`for await` にすると detokenizer の部分 UTF-8 持ち越し（ADR 0084 決定 4）が「push すると確定した
文字列だけ出る」形と直結し、`break` / `return()` で自然に閉じる。

MUST（`break` の穴）: 中断された iterable は `return()` 経由で `finally` に入るが、そこで
**論理長は成功した run のぶんだけ既に進んでいる**。`pendingToken` を正しく残さないと次ターンで
1 token 落ちる。`for await` を採る以上、この経路の門は必須（段 3 の合格線）。

MUST: sequence は「generate 1 回ぶん」を `createOperationChain`
（`packages/models/src/concurrency/serial.ts`）で直列化する — 他 3 家族と同型。**自前ロックは
作らない**（ランタイム側の single-flight が既に 2 本目の発行を拒否する）。

### 3. `GenerationContext` は sequence の内側に隠す（裁定 2）

製品面に素の `GenerationContext` を出さない。「最大 1 token の未 commit frontier」（決定 4）は
消費者から見えなくなる。検収経路（`generateGreedy` / e2e）は内部ヘルパとして context を直接
使い続けるので、**公開面が二重にならない**。

### 4. 多ターンは `pendingToken` を連結した prefill

`GenerationContext` は常に**最大 1 token の未 commit frontier**を持つ（K token 生成後の
`pastLength = T + K − 1` — `generateGreedy` が `maxNewTokens - 1` 回しか decode しないため。
レビュー CG5-1 で追認）。したがって次ターンは、**`pendingToken` を新しい user turn の token 列の
先頭に連結して prefill する**。余分な「commit だけの run」を増やさずに frontier を KV へ収める形で、
これが決定 3（context を隠す）を成立させる要でもある。

MUST: `rewind` は使わない — sliding スロットを含む context は**全拒否**（ADR 0066 追記 2）。
編集・分岐は「新しい context + token transcript の replay」が正。

### 5. cancel は `AbortSignal` が正（`onEvent` の throw を中断手段にしない）

`GenerationRequest.signal` を受け、段の境目で検査して `signal.reason` を**包まずそのまま
throw** する。流儀の前例は構築経路の `AnimaPipelineOptions.signal`。backlog later の「生成ループの
AbortSignal 中断席（需要待ち）」は **LLM がその需要**なので、本 ADR で席を開ける。

### 6. 出口は最終行 logits・sampling はホスト維持（裁定 4 — topk 製品グラフは棄却）

製品グラフの出口を `logits[1,1,V]`（**最終行のみ**）にする。`last_row` 行選択 → 1 行 lm_head の
配線は `tools/export-recipes/gemma4/export_token.py` の `TokenOnlyChunkWrapper` に既にあり、
**argmax を外すだけ**で届く。ADR 0068 決定 4 は全語彙 logits の readback を「グラフ出力に logits
を宣言した場合の opt-in として残す（欄を消さない）」と書いており、却下したのは **GPU 側 full
sampling** の方である。よって本決定は**再裁定を要さず**、ADR 0068 には追記 6 を置く。

**topk 製品グラフを採らない根拠**（棄却理由の記録）:

1. **`k ≤ 63` がモデル自身の推奨既定に 1 だけ足りない**。実装上限は ADR 0068 追記 2 の
   `8·W·(k+1) ≤ maxComputeWorkgroupStorageSize`（W=32・WebGPU 既定 16384B）で **k ≤ 63**、
   一方 gemma-4-E2B-it の `generation_config.json` の推奨既定は **`top_k: 64`**（実資産の実測 —
   ドラフト §0.2-8）。ADR 0068 自身の却下文が「topk k 本の readback（k ≤ 64）」を前提にしていた
   のに実装が 63 に着地したので、この 1 の差がそのままモデル既定と噛み合わない。
   「実用上 63 で十分」と言い張れる形ではない。
2. **exporter の停止点は handler ではなく `operator.getitem`**（ADR 0068 追記 3 の実測）—
   aten handler を足しただけでは道が開かない。
3. **repetition penalty / logit bias / full-vocab nucleus は全語彙が要る**（top-63 の外の softmax
   質量をホストが知らない）。chat 用途でこれらを持たない選択は現実的でない。
4. **コストが壁に対して無視できる**。1 MiB の readback は同じ submit に相乗りするのでフェンスは
   増えず、262,144 要素の JS 走査込みで **0.3〜0.6ms 級の見積り**。decode 32.5ms/token
   （ADR 0082）に対して数%で、フェンス床 ≈11ms が支配へ戻った現在でも桁は変わらない。
   **prefill の読み戻しはむしろ減る**: 現行の logits opt-in 系列は `[1,M,V]` を返すので chunk 32 で
   32 MiB、最終行だけなら **1 MiB**。

MUST: **sampling / RNG はホスト維持**（ADR 0068 決定 4 の再確認 — GPU 側は argmax / topk のみ）。

### 7. sampler — 置き場・RNG 契約・既定値の所有者（裁定 7）

- **置き場** = `packages/models/src/generation/sampler.ts`（`greedy.ts` と同じ「パイプライン
  非依存の共通処理」の位置づけ）。初版の射程 = 温度 / top-k（**上限なし** — 決定 6 の出口だから）/
  top-p / repetition penalty / logit bias / seed。
- **RNG** = `packages/models/src/anima/random.ts` の splitmix64 の流儀を再利用する。
  MUST（前例そのまま）: その doc が「torch の `randn` とは別物・同じ seed でも同じ列にならない」と
  書いているのと同じ理由で、**HF の sampling 出力との token 列 parity は取れない**。parity 門は
  温度 0（= greedy）で採り、sampler 自体は自前 fixture + 分布門で縛る。
- **既定値は配布形が宣言する**: `generation_config.json` の推奨（gemma-4-E2B-it =
  temperature 1.0 / top_k 64 / top_p 0.95）を dist が `pipelineConfig` へ焼き、リクエストが
  `sampler` を省略したときの既定にする。前例は anima の `scheduler.type`（ADR 0078 — 配布形が
  既定を宣言し `AnimaGenerateRequest.sampler` で上書きする形）。**最終行 logits 出口だからこそ
  `top_k: 64` がそのまま表現できる**（決定 6 の裏返し）。
  NOTE: 低層（`sampler.ts` 単体・`generateGreedy`）の既定は greedy のままで、これは
  「sampler 未宣言 = 温度 0 の縮退形」という位置づけである。parity 門はこの経路で生き続ける。

### 8. 停止 token は集合（モデル宣言由来）

停止条件は単数の EOS ではなく**集合**である: gemma-4-E2B-it の `generation_config.json` は
`eos_token_id = [1, 106, 50]`（`<eos>` / `<turn|>` / `<|tool_response>` — 実資産の実測）。
`GenerationProgram.stopTokens` で受け、**sampling の結果に対して**判定する（sampler と同じ段）。

MUST: EOS 集合と chat 形式は**同じ配布 digest set から来る**（ADR 0084 決定 5）— 別々の場所から
拾うと片方だけ古くなる。

### 9. `generateGreedy` は内部ヘルパへ格下げする（裁定 8 — breaking・今波）

barrel（`packages/models/mod.ts`）から export を外し、parity 検収用の内部ヘルパにする。
「当面は公開のまま残して次の breaking 波で外す」は採らない。既に公開済みの面なので**破壊的変更**
として limitations とリリースノートに載せる。**消えるのは公開 export だけ**で、既存 golden の
token 列を突き合わせる parity 門は経路として残る。

### 10. 容量超過は専用の型で落とす

full スロットの `P + Q ≤ C` 超過は今日「汎用メッセージで fail loudly」。sequence は専用の型
（`GenerationCapacityError` 相当）で落とし、**「会話の切り詰めはホストの責務」**を limitations に
明文化する（lens-llm L-9 b の消化）。

## 検討した代替案

- **anima 流 `onEvent` コールバック**（家風に完全一致 — await する / 例外を握らない / throw が
  step 粒度の中断手段）: 放置された async generator が直列化鎖の席を握る危険が無いのが利点で、
  リポ初の公開 async generator を作らずに済む。却下の理由は決定 2 のとおり「token 列は進捗では
  なく値」— streaming 消費者に自前キューを組ませる形になる。AbortSignal を cancel の正に据えた
  （決定 5）ので、throw を中断手段にする必要も同時に消えた。
- **関数 1 本の拡張（`generateStream(spec)`・sequence 型を作らない）**: 追加型ゼロで最小だが、
  `GenerationContext` を製品面に露出する形になり、`pendingToken` の管理が呼び手へ漏れる。
  「直前 assistant の最後の token が履歴から 1 個落ちる」事故を消費者側で再生産するので却下。
- **topk 製品グラフ（argmax → topk・k 固定）**: readback は 512B 級で済むが、決定 6 の 1〜3 の
  とおり。**GPU 側 full sampling / nucleus** は ADR 0068 が正面から却下済み（採るなら再裁定要）。

## Consequences

- 実装は backlog now の実行計画（6 段）。**段 0（契約固め）は本 ADR + ADR 0084 / 0085 +
  ADR 0068 追記 6 で完了**（コード 0 行 — ADR がイベント契約そのもの）。
- **破壊的変更 1 件**: `generateGreedy` の公開 export 削除（決定 9）。limitations が消費側の doc。
- **段 1b の再 export は案 α**（裁定 5）: PLE 外出し（ADR 0085）と最終行 logits 出口を**同じ
  再 export に載せ、製品グラフを 1 系列にする**。3.7GiB 系列の再 export が 1 回で済む代わりに、
  ホスト PLE loader が e2e の前提になる。既存 2 系列（logits opt-in / token-only）は検収 fixture
  として残す。案 β（logits 出口だけ先行・PLE は後）は再 export 2 回 + 段 2〜4 の fixture 採り直しに
  なるので採らない。
- **配布対象は gemma4 E2B のみ**（裁定 10）— ライセンス門を待つ間に minicpm5 で配布経路だけ先行
  させる案は採らない。公開は ADR [0065](0065-exporter-core-recipe-split.md) 決定 7（stage 6 =
  upstream revision 単位のライセンス interview）後で、技術側の完成と独立に止まりうる。
- **本波では採らないので現時点の再裁定を要さない 4 件**（採るときは各々再裁定が要る — ドラフト §9）:
  1. `rewind` の利用 — sliding を含む context は全拒否（ADR 0066 追記 2）。緩和は compaction
     実装と対で裁定する。
  2. continuous batching / batch > 1 / 複数シーケンス — ADR 0066 決定 8 のスコープ外のまま。
  3. GPU 側 full sampling / nucleus — ADR 0068 が却下済み（決定 6 の MUST がこれを維持する）。
  4. RoPE 表のホスト生成（lens-llm L-9 c）— ADR 0067 決定 4「RoPE は attention op の外・
     エクスポータがグラフに焼く」と衝突する。
- 射程外（本波でやらない）: decode 性能そのもの（ADR 0082 で消化済み・以降はレンズ L-7 / L-12 の
  再評価 = perf-ledger）/ KV の f16 席 / topk の exporter 配線（決定 6 を採る限り実需が立たない）。
