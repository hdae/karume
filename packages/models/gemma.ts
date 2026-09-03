/**
 * `@karume/models/gemma` — Gemma ファミリ（gemma4 = 文字列 → 文字列の生成）のサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link Gemma4Pipeline.fromPretrained} /
 * {@link Gemma4Pipeline.fromAssets}）/ 会話する（`chat`）/ 自分で回す（`sequence` +
 * {@link gemma4ChatPrompt} + tokenizer の復号）/ 解放する（`dispose`）。
 *
 * 生成の語彙（`GenerationEvent` / `GenerationStop` / `SamplerSpec` / `GenerationCapacityError`）
 * は**パイプライン非依存**だが、今のところ触れるのはこのファミリだけなので barrel とこの面の
 * 両方から出す。`GenerationProgram` / `GenerationSequence` を**作る**関数は出さない — 入口は
 * パイプラインだけで、静的配線を迂回して組める面を作らない。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

export { Gemma4Pipeline } from "./src/gemma/pipeline.ts";
export type {
  Gemma4Assets,
  Gemma4ChatOptions,
  /** `chat` の停止理由（sequence 層の理由 + この層でしか判定できない `stop-string`）。 */
  Gemma4ChatStop,
  Gemma4ChatStream,
  /** `estimateSessionMemory` が見積る生成の形（容量 / chunk 長 — どちらも実行時ノブ）。 */
  Gemma4EstimateOptions,
  Gemma4FromPretrainedOptions,
  Gemma4PipelineOptions,
  /** prefill の進捗 1 通（`chunk / chunks` — `onPrefill` が受ける）。 */
  Gemma4PrefillProgress,
  /** `sequence()` の指定（この会話が確保する容量）。 */
  Gemma4SequenceOptions,
} from "./src/gemma/pipeline.ts";
/**
 * 多ターンの会話を持ち回る中間層（`chat` と `sequence` の間 — ADR 0083 追記 2026-09-02）。
 *
 * `chat` は 1 ターン = 1 sequence で過去 turn を毎回描き直し（会話が伸びるほど prefill が
 * O(n²)）、`sequence` は token id と KV の寿命を呼び手へ渡す。{@link Gemma4ChatSession} は
 * その落差だけを持つ — KV を継ぎ、容量が足りないターンは**送る前に**
 * {@link Gemma4ChatSessionOptions.onOverflow}（既定 {@link dropOldestTurns}）で切り詰める。
 */
export { dropOldestTurns, Gemma4ChatSession } from "./src/gemma/chat-session.ts";
export type {
  Gemma4ChatOverflow,
  Gemma4ChatOverflowPolicy,
  /** セッションがパイプラインから読む面（{@link Gemma4Pipeline} がそのまま満たす）。 */
  Gemma4ChatSessionHost,
  Gemma4ChatSessionOptions,
  Gemma4ChatTurnOptions,
} from "./src/gemma/chat-session.ts";
/** 配布形が宣言する静的配線（`karume.json` の `pipelineConfig` — ADR 0038 §1）。 */
export type { Gemma4DefaultSampler, Gemma4PipelineConfig } from "./src/gemma/config.ts";
/**
 * **このパッケージ版が検証した取得元**（`hdae/karume-gemma4-e2b` の pin 済み commit SHA —
 * ADR 0073）。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン** — {@link Gemma4Pipeline.fromPretrained}
 * の第 1 引数へそのまま渡す。再現性を自分で固定したい場合は、この定数ではなく自分の
 * `{ repo, revision }` を書く（`fromPretrained` に既定は無い）。
 */
export { GEMMA4_CURRENT } from "./src/gemma/config.ts";
/**
 * 生の宣言（`unknown`）→ 検証済みの {@link Gemma4PipelineConfig}（失敗は fail loudly）。
 *
 * **`fromAssets` を使う消費者のための口**である。`fromPretrained` は内部でこれを通すので呼ぶ
 * 必要は無く、要るのは `karume.json` を自分で読んで {@link Gemma4Pipeline.fromAssets} へ渡す側
 * （手元にあるのは hub の `ModelEntry.pipelineConfig` = `Record<string, unknown>` なので、
 * この口が無いと `as` で被せるか 3 つの数を二重持ちするしかない）。
 */
export { parseGemma4PipelineConfig } from "./src/gemma/config.ts";
/**
 * 配布形が宣言する RoPE のパラメータ（`pipelineConfig.rope` — 層種別 2 本）。
 *
 * 表は配布形に無く、cos / sin は chunk ぶんだけホストが作る。式の正本は
 * `src/gemma/rope.ts`（exporter 側は鏡像）。
 */
export type { Gemma4RopeLayerSpec, Gemma4RopeLayerType, Gemma4RopeSpec } from "./src/gemma/rope.ts";
/**
 * 位置列 → RoPE のグラフ入力 4 本（`rope_{sliding,full}_attention_{cos,sin}` の
 * `[1, rows, headDim]` f32）。
 *
 * `Gemma4Pipeline` は内部でこれを派生入力の席へ差すので呼ぶ必要は無い。要るのは**製品グラフを
 * 自分で回す**側（低レベル面より下 — 実 Session に自分で run を出す検収・計測の経路）で、この口が
 * 無いと式を写して持つしかない（写した時点で「同じ式を 2 回書いて比べただけ」になる）。
 */
export { gemma4RopeInputs } from "./src/gemma/rope.ts";

/**
 * 会話 → token id 列（`<bos>` 込み・末尾は生成プロンプト）。素の会話だけを受け、tools /
 * thinking / 未知 role は fail loudly で拒否する（ADR 0084 決定 5）。
 *
 * `chat` は内部でこれを通すので、要るのは低レベル面（`sequence`）を自分で回すときだけ。
 */
export { gemma4ChatPrompt } from "./src/gemma/text/chat.ts";
/**
 * 新しい発話 1 つ → **会話の続きとして描き足す差分 token 列**（多ターンを `sequence` で回すときの
 * 2 ターン目以降の入口 — 過去 turn は context の KV にあるので描き直さない）。
 *
 * MUST: 前 turn を閉じる `<turn|>` は含めない（`GenerationSequence` の `pendingToken` が前置する
 * — turn-local 契約の正本は `renderGemma4ChatTurn` の doc、成立は `gemma_chat_test.ts` の門）。
 */
export { gemma4ChatTurn } from "./src/gemma/text/chat.ts";
export type { Gemma4ChatMessage, Gemma4ChatRole } from "./src/gemma/text/chat.ts";

/**
 * 資産から組んだトークナイザ（`Gemma4Pipeline.tokenizer`）。低レベル面で受け取った token id を
 * 文字列へ戻す口（`decode` / `createDetokenizer`）がここにある。
 *
 * 値としては公開しない — 入口は `Gemma4Pipeline.fromAssets` で、資産の突合を迂回した半端な
 * トークナイザを作れる面にしない（ADR 0008）。
 */
export type { GemmaDecodeOptions, GemmaTokenizer } from "./src/gemma/text/tokenizer.ts";
export type { StreamingDetokenizer } from "./src/text/detokenizer.ts";

/**
 * 生成の語彙（ADR 0083）。`sequence()` を自分で回すときに要る型と、「会話が入り切らない」を
 * 他の失敗と読み分けるための例外（決定 10 — **低レベル面では**会話の切り詰めはホストの責務。
 * 高レベル面 {@link Gemma4ChatSession} は注入可能な既定ポリシーを持つ — 同 追記 2026-09-02）。
 */
export { GenerationCapacityError } from "./src/generation/sequence.ts";
export type {
  GenerationCapacityConstraint,
  GenerationCapacityDetail,
  GenerationEvent,
  GenerationRequest,
  GenerationSequence,
  GenerationStop,
  GenerationStream,
} from "./src/generation/sequence.ts";
export type { GenerationProgram } from "./src/generation/program.ts";
export type { SamplerSpec } from "./src/generation/sampler.ts";
