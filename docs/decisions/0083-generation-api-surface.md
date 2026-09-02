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

## 追記

- 2026-08-31（公開面レビューの消化 — 裁定の原文 = `.claude/reviews/2026-08-31_182ced7/`）:
  実走した消費者が「面から読めない数」を推定で埋めていた 3 点と、公開面の書き換え口 1 点を閉じた。
  いずれも決定 1〜10 の設計は不変で、**読み口を足すか、書ける口を塞ぐか**である。
  - **`GenerationStop.tokens`**（決定 2 の停止イベント）: そのターンが生成した token 数。
    MUST: `eos` の停止 token も 1 個として数える — 抽選 1 回 = run 1 回なので、この数がそのまま
    生成に費やした run 数と一致し、`tok/s` を**再エンコード無し**で書ける（それがこの欄の目的）。
    本文だけの数が要るなら `reason === "eos"` のとき 1 引く。従来は消費者が本文を再 encode して
    token 数を推定していた（tokenizer の往復が恒真でない以上、推定は原理的に外れうる）。
  - **`GenerationSequence.used`**（決定 1 の「可変な寿命」の読み口）: この会話が既に占めている
    論理位置の数。MUST: **導出値**（`context.pastLength` + 未 commit frontier — ADR 0066 決定 6 の
    二重簿記の禁止）で、独立した counter は持たない。次のターンが通るかは
    `used + prompt.length + maxNewTokens - 1 ≤ capacity` かつ
    `used + prompt.length + maxNewTokens - 2 < maxPosition` で事前に判定できる。
  - **`GenerationCapacityError` の構造化欄**（決定 10 の専用型を「読み解かせない」形へ）:
    `constraint`（`capacity` / `maxPosition` のどちらを踏んだか）/ `pastLength` / `promptLength`
    （**`pendingToken` 連結後**の実数 — 呼び手が渡した `prompt` より 1 多いことがある）/
    `requestedNewTokens` / `limit` / `maxNewTokens`（この prompt のまま今なら通る上限・負値は
    「何 token 溢れているか」）。専用型を持つ意味は文言ではなく**切り詰めの計算に要る数**なので、
    それを欄で渡す。
  - **公開 `GenerationProgram` を読み口だけへ絞る**（決定 1 の「program は不変」の強制）:
    生成ループが読む全欄は内部型 `GenerationWiring` へ分離し、公開面（`Gemma4Pipeline.program`）が
    出すのは自分で `sequence()` を回すときに要る数（`chunkLength` / `capacity` / `maxPosition` /
    `vocabSize` / `stopTokens`）だけにした。グラフ入出力の名前・記号束縛・`derivedInputs` を
    公開すると、①消費者が読んでも使い道が無い（配線の相手である Session は公開面に無い）
    ②`derive` の差し替えや `bindings` の改変が公開面から書ける — **検証済みであること**が
    `GenerationProgram` の意味そのものなので、書ける口は意味を壊す。面は凍結して返し、
    `stopTokens` は**凍結コピー**にする（同じ配列を出すと消費側の `sort()` / `length = 0` が
    生成ループの停止集合を書き換え、「EOS で止まらない生成」として沈黙劣化する）。
  - **破壊的変更（未リリース面）**: `GenerationStop` の欄追加・`GenerationCapacityError` の
    コンストラクタ 2 引数化・`GenerationProgram` からの配線欄の消滅・`dispose()` 後の `generate`
    が**同期 throw** になったこと。消費側の doc は `docs/limitations.md`。

- 2026-09-02（同じ公開面レビューの第 2 波 — 決定 1 の「program は不変」と同じ規律を**要求の側**へ
  広げた）: 面の設計は不変で、**呼び手が握ったままの object を後から書き換える口**を 2 つ塞ぐ。
  - **`SamplerSpec.logitBias` は `Map` ではなくタプルの配列**（`readonly [token, bias][]` —
    決定 7 の sampler の受理集合）。この指定は JSON を通る経路（配布形の宣言・ホスト側の設定
    保存・ログ）と worker 境界を跨ぐが、`Map` は `JSON.stringify` が `{}` へ潰すので、往復した
    指定は**黙って空の bias** になる（例外は出ず「効いていないノブ」としてだけ現れる）。
    MUST: 同じ token を 2 度書いたら fail loudly — `Map` は後勝ちで畳んでいたが、加算の面で
    畳むと「2 つ書いた bias の片方だけが効く」形になり、足すのか上書きするのかを呼び手が
    読めない。**破壊的変更（未リリース面）**。
  - **要求は発行時に写す**（`GenerationSequence.generate` / `Gemma4Pipeline.chat`）: `prompt` の
    複製・`maxNewTokens` / `signal` の束縛・sampler 指定のスナップショット（`logitBias` は要素の
    タプルまで写す）を**発行の同期部分**で済ませる。async generator の本体は最初の `next()` まで
    走らないので、写さないと「受理集合を検査した値」と「実際に流す値」が別物になり得た — 検査後に
    `prompt` へ語彙外 id を足す / 走行中に `maxNewTokens` を伸ばす / `logitBias` を差し替える、の
    どれも例外にならない。複製はレイヤごとに 1 箇所（`generate` が prompt・`createSampler` が
    sampler 指定）で、`samplerDistribution` は 1 回の呼び出しの中でしか spec を読まないので写さない。
    観測できる変化は「発行後の書き換えが効かなくなる」ことだけで、消費側の doc は
    `docs/limitations.md`。

- 2026-09-02（同じ第 2 波の続き — **要求ごとの停止条件**と**一括受け取り**）: 決定 8 の停止集合は
  「配布形が宣言した EOS」しか無く、ターンごとに止め方を変える術が公開面に無かった（消費者は
  detokenize しながら自分で `break` する = 停止文字列の跨ぎと未 commit frontier の扱いを再実装
  することになる）。面の設計は不変で、**停止条件を 2 層に分けて要求側へ開く**。
  - **停止 token は sequence 層**（`GenerationRequest.stopTokens`）: 配布形の集合との**和集合**で
    判定する（配布形の EOS は常に効く）。`GenerationStop` に枝 `stop-token` を足して読み分ける —
    両方の集合に居る id は `eos` で閉じる（配布形の終端記号としての意味が優先する）。語彙外・
    重複は**同期に** fail loudly: 停止 token は「出力に現れない id」なので、間違っていても生成は
    普通に完走し、出力が伸び続けることでしか気づけない。指定は発行時に写す（上の追記と同じ規律）。
  - **停止文字列は chat 層**（`Gemma4ChatOptions.stopStrings`）: sequence は token id しか扱わない
    ので、文字列の判定は復号の**後**にしか置けない（1 つの停止文字列が複数 token に割れることも、
    1 つの token が停止文字列の末尾と次の本文をまたぐこともある）。実体は
    `src/text/detokenizer.ts` の `createStopStringFilter`（ファミリ非依存）で、`byte_fallback` の
    run 持ち越しとは**別の状態機械**である。
  - **保留（holdback）の規則**: 確定した文字列でも、**停止文字列の接頭辞になっている末尾**だけは
    出力を保留する。固定長（最長の停止文字列 − 1 文字）で保留すると止まらないターンでも描画が
    常に遅れるので、接頭辞から外れた時点でまとめて流す（例: 停止 `"END"` に対し `"E"` `"N"` `"X"`
    は 3 片目で `"ENX"` がまとめて出る）。一致したら**停止文字列の手前まで**を流し、停止文字列
    そのものとその後ろは流さない。止まらずに終わったターンは保留ぶんを最後に流す（1 文字も
    落とさない — 保留は判定のための遅延であって切り詰めではない）。
  - **早期終了の畳み方**: 停止文字列で止めるときはイベント列から `return` で抜ける（`for await`
    の脱出が `return()` を呼ぶ）。既存の `break` 中断とまったく同じ後始末で、会話は成功した run の
    ぶんだけ進み、最後の token は未 commit frontier（`pendingToken`）に残る — 停止 token で
    止めたときと同じ状態である。停止理由は chat 層で `stop-string`（`Gemma4ChatStop`）へ差し替え、
    `tokens` は内側の `done` の数をそのまま運ぶ（この層で数え直さない = 二重簿記の禁止）。
  - **一括受け取り `Gemma4ChatStream.text()`**: 汲み切って連結した 1 本を返す。**1 つの
    ストリームは 1 通りにしか消費できない**（反復と `text()` の併用も、2 度の反復も**同期に**
    throw）— 生成は 1 度しか走らないので、2 通り目には「残り」しか流れず、例外にならない
    取り違えになる。メソッドを `chat` の隣に増やさず stream 側に置いたのは、停止理由（`done`）と
    同じ 1 つの返り値にぶら下がる観測口だからである。
  - **破壊的変更（未リリース面）**: `GenerationStop` の union に `stop-token` が増えた・
    `Gemma4ChatStream.done` の型が `GenerationStop` から `Gemma4ChatStop` になった。消費側の doc は
    `docs/limitations.md`。

- 2026-09-02（同じ第 2 波の続き — **中間層 `Gemma4ChatSession`** と、それに伴う**決定 10 の改訂**）:
  実走した消費者の指摘は「普通の stateful chat が欲しいだけなのに、`chat()` は毎回全履歴を
  再計算し、効率化しようとすると急に token id / `<bos>` / 未 commit frontier の世界へ落とされる」。
  低レベル面（決定 1〜5）も `chat()` も設計は不変で、**その落差だけを持つ層**を足す
  （`packages/models/src/gemma/chat-session.ts`）。素材は既存の公開面だけ — `sequence()` の KV 継続 /
  `gemma4ChatPrompt` と `gemma4ChatTurn` / `GenerationSequence.used` /
  `GenerationCapacityError` の構造化欄 / `chat()` の停止条件と `Gemma4ChatStream` — なので、
  消費者は同じものを自分で書ける（写経見本だった `examples/gemma4/main.ts` の中身がここへ
  昇格し、example は「普通の chat」の書き方に戻った）。
  - **決定 10 の改訂**: 「会話の切り詰めはホストの責務」は**低レベル面についてはそのまま**
    （`sequence()` / `chat()` は今も `GenerationCapacityError` を投げて終わる）。~~高レベル面でも
    同じ~~ ← 打ち消し: `Gemma4ChatSession` は**注入可能な既定ポリシー**
    （`Gemma4ChatSessionOptions.onOverflow`）を持つ。既定は `dropOldestTurns` = **最古の
    user / assistant の対を落とす**（system 発話は残す・片方だけ落とすと「答えだけ」「問いだけ」の
    壊れた文脈になるので対を単位にする）。ホストが throw する関数を渡せば従来の意味論に戻る。
    「切り詰めない」を既定にしなかったのは、会話 UI で**必ず**要る打ち手を全消費者に再実装させる
    形だからで、方針そのものは差し替えられるべきという判断は決定 10 のまま残っている。
  - **判定は事前**（例外を待たない）: 各ターンの prompt を組んだ直後に
    `used + prompt + maxNewTokens - 1` を上限（`capacity` と `maxPosition` の小さい方）と比べ、
    超えていれば送る前に `onOverflow` を回す。`used` を公開したのはこの判定のためである
    （追記 2026-08-31）。走行中に踏む `GenerationCapacityError` の捕捉は**安全網として残さない** —
    事前判定が正しい限り到達しないので、掴んで再試行すると事前判定のずれが黙って隠れる。
  - **収束の保証**: 再試行が続くのは**履歴が縮んだとき**だけ（縮まない結果を返したポリシーは
    `GenerationCapacityError`）。履歴の件数は非負整数なので、試行回数は入口の発話数を超えない —
    その不変条件が破れたことは `#prepare` の門が名指しで落とす（回数の定数を置かない）。
  - **KV を継ぐ条件**: 直前のターンが**配布形の EOS で閉じた**ときだけ差分（`gemma4ChatTurn`）で
    継ぐ。max-tokens / 停止文字列 / 中断 / 失敗で閉じたターンの後ろは model turn が閉じていないので、
    sequence を捨てて履歴から描き直す（決定 4 の turn-local 契約をそのまま門にした形）。ターンの
    締めで履歴へ積むのは**実際に流した本文**だけで、1 文字も出なかったターンはその発話ごと履歴から
    外す（答えの無い問いを残さない）。
  - **1 セッション = 1 生成**: 2 本目の `send` は**同期に** throw する（`chat()` のように順番待ちに
    しない）。履歴は 1 本の会話として順に積まれるので、2 本が同じ履歴を押すと「答えの無い問い」を
    挟んだ会話が KV へ入る — 例外にならない取り違えなので口の側で塞ぐ。代償として、発行した
    stream を汲まずに捨てるとセッションはそのターンのまま止まる（締めが列の終端で走るため）。
  - **入口はコンストラクタ**（`new Gemma4ChatSession(pipeline, options)`）: `Gemma4Pipeline` に
    `createChatSession()` を生やすと pipeline → chat-session → pipeline の相互 import になる
    （`decodeChatChunks` / `chatStreamOf` を共有するため）。依存を一方向に保つ方を採り、
    セッションが読む面は `Gemma4ChatSessionHost`（tokenizer / program / defaultSampler /
    `sequence()`）へ絞った — この層が公開面より内側を 1 つも使っていないことが型で読め、
    実 GPU 無しの門（`tests/gemma_chat_session_test.ts`）もそこから来る。
  - **今回入れないもの**: モデルに要約させる compact（`onOverflow` の実装の 1 つとして後から
    足せる席はある）。**窓（`capacity`）を広げた後に再検討する** — 今の 640 では要約自体が
    入らない。
