/**
 * 計測に使う固定 prompt 4 種（chat のメッセージ列を組むだけ — GPU も tokenizer も触らない）。
 *
 * 投機の取り分は**文脈で変わる**（受理率が変わる）ので、1 つの prompt で測った倍率は他の用途を
 * 代表しない。4 種はその幅を狙って選んである:
 *
 * - `extract` — 長文脈からの抽出（文書の語をそのまま写す ⇒ 受理率が最も高く出る側）
 * - `summarize` — 同じ長文脈からの要約（自分の言葉で書く ⇒ 抽出より受理率が落ちる）
 * - `dialogue` — 3 発話の対話（会話の続き = 実用の形）
 * - `freeform` — 短い prompt からの創作（文脈の手掛かりが最も薄い側）
 *
 * MUST: 素材は**リポの中の git 追跡下の文書**から採る（`outputs/` は追跡外なので、そこから読むと
 * 「同じ台本を回しても別の prompt になる」）。MUST: 切るのは**段落境界**（上限の手前の最後の空行）
 * — 文字数で切ると文の途中で終わり、生成が同じ数語を繰り返す退化列になる。どちらも
 * `tools/export-recipes/gemma4/export_drafter.py` の golden 素材と同じ規則である（同じ規則で
 * 焼いた drafter を、同じ規則で組んだ prompt で測る）。
 *
 * `--warm`（モードごとに 1 本の会話を伸ばす走行）が 2 本目以降のターンで追記する発話列は
 * {@link warmFollowUps}（ターンごとに違う依頼 — 理由はそこの doc）。
 *
 * MUST: 全モジュール副作用ゼロ（文書はモジュールスコープで読まない — 読むのは
 * {@link buildWorkload} が呼ばれたときだけ）。
 */

import type { Gemma4ChatMessage } from "../../packages/models/gemma.ts";

export type WorkloadName = "extract" | "summarize" | "dialogue" | "freeform";

/** 受け付ける綴り（CLI の検査と使い方の表示が同じ列を読む）。 */
export const WORKLOAD_NAMES: readonly WorkloadName[] = [
  "extract",
  "summarize",
  "dialogue",
  "freeform",
];

/** 文書を素材に取るワークロードの綴り（{@link DOCUMENT_QUESTION} の鍵でもある）。 */
export type DocumentWorkloadName = "extract" | "summarize";

/** 文書を素材に取るワークロード（`--document-chars` が効くのはこの 2 つだけ）。 */
export const DOCUMENT_WORKLOADS: readonly WorkloadName[] = ["extract", "summarize"];

/**
 * 文書系かどうか（判定の正本は {@link DOCUMENT_WORKLOADS} 1 本）。
 *
 * 型述語にしてあるのは、綴りを条件式へ写さずに `DOCUMENT_QUESTION` の鍵として使えるようにする
 * ためである — 一覧と条件式が別に育つと「一覧には入っているが分岐に入っていない」ができる。
 */
export const isDocumentWorkload = (name: WorkloadName): name is DocumentWorkloadName =>
  DOCUMENT_WORKLOADS.includes(name);

/** 素材の文書（リポ内・git 追跡下）。`export_drafter.py` の長いケースと同じファイル。 */
const DOCUMENT_URL = new URL("../../tools/exporter/README.md", import.meta.url);

/** 文書の**前**に置く指示（`export_drafter.py` の `DOCUMENT_INSTRUCTION` と同文）。 */
const DOCUMENT_INSTRUCTION = "Read the following project document.\n\n";

/** 文書の後ろに置く依頼文（ワークロード別 — ここが `extract` と `summarize` の唯一の差）。 */
const DOCUMENT_QUESTION: Readonly<Record<DocumentWorkloadName, string>> = {
  extract: "\n\nList every shell command that appears in the document, verbatim, one per line.",
  summarize: "\n\nSummarize this document in 10 bullet points.",
};

/** 自由文のケース（`export_drafter.py` の `SHORT_PROMPT` と同文 — golden の short-en と同じ会話）。 */
const FREEFORM_PROMPT =
  "Write a short story (about 300 words) about a lighthouse keeper who discovers a message " +
  "in a bottle.";

/** 対話ケースの 1 発話目。 */
const DIALOGUE_QUESTION =
  "I am writing a small WebGPU inference library in TypeScript. What are the three biggest " +
  "performance pitfalls I should watch for?";

/**
 * 対話ケースの assistant 発話 — **定数として焼いた固定の返答**である。
 *
 * MUST: モデルに生成させた文を持ってこない。prompt が走行ごとに変わると、受理率も壁も
 * 「文脈が違うだけ」で動き、ABBA でも中央値でも打ち消せない。3 発話目（続きの依頼）が
 * 参照する「2 点目」がここに固定されていることが、このケースの再現性そのものである。
 */
const DIALOGUE_ANSWER = [
  "The three that cost the most are all about how often the CPU and the GPU wait for each other.",
  "",
  "1. Per-dispatch fence waits. Submitting one command buffer per operation and awaiting its " +
  "completion turns every kernel into a round trip. Batch a whole forward pass into one " +
  "submission and fence once at the end.",
  "",
  "2. Buffer re-allocation. Creating storage buffers inside the hot loop makes the driver zero " +
  "fresh memory and rebuild bind groups on every step. Allocate once, pool by size class, and " +
  "keep bind groups alive as long as their buffers live.",
  "",
  "3. Readback stalls. Mapping a result buffer drains the queue, so a single readback per step " +
  "serialises the whole pipeline. Keep intermediates on the device and read back only what the " +
  "host must actually branch on.",
  "",
  "Measure with timestamp queries before changing anything: the bottleneck is usually not the " +
  "kernel you suspect.",
].join("\n");

/** 対話ケースの 3 発話目（固定の返答の「2 点目」を指す）。 */
const DIALOGUE_FOLLOW_UP = "Expand on the second point with a concrete example.";

/**
 * `--warm` の 2 本目以降が追記する user 発話（自由文）— **ターンごとに違う**依頼である。
 *
 * MUST: 同じ発話を繰り返さない。同じ依頼をもう 1 度流すと model は前のターンの答えを写し始め、
 * drafter の受理率が跳ねる（実測 1.63 → 3.9 tok/cycle・docs/research 2026-09-09 §6.5）。warm は
 * 「投機が負ける課題のまま、ゲートだけを暖める」ために在る口なので、写しが起きた走行は測りたい
 * ものを測っていない。主題も形も**文体**（物語 / 説明 / 手順 / 報道 / 皮肉 / 抒情 / 演説 …）も
 * 変える — 同じ文体の 300 語が KV に積み上がるほど drafter は n-gram を拾いやすくなる。長さの
 * 指定だけ {@link FREEFORM_PROMPT} と同じ級に揃えてある（詩だけ行数）。
 */
const FREEFORM_WARM_FOLLOW_UPS: readonly string[] = [
  "Write a short story (about 300 words) about a bakery that opens only during thunderstorms.",
  "Explain to a ten-year-old (about 300 words) why the sea is salty but most rivers are not.",
  "Write a product review (about 300 words) of an umbrella that only opens when it is not " +
  "raining.",
  "Write a letter (about 300 words) from a retired cartographer to the island she never " +
  "finished mapping.",
  "Write step-by-step instructions (about 300 words) for teaching a cat to answer the doorbell, " +
  "in the tone of a serious appliance manual.",
  "Write a dialogue (about 300 words) between a locksmith and a customer who has forgotten " +
  "what the key opens.",
  "Write a newspaper report (about 300 words) on a town that voted to set its clocks back by " +
  "one hour every Monday.",
  "Write a poem (about 20 lines) about the last train of the night leaving an empty station.",
  "Write a persuasive speech (about 300 words) arguing that staircases should count as public " +
  "art.",
  "Write a diary entry (about 300 words) by a night-shift museum guard who suspects that one " +
  "painting changes.",
  "Write a travel-guide entry (about 300 words) for a village whose streets are renamed every " +
  "spring.",
  "Write a folk tale (about 300 words) explaining why the moon owes the sea a favour.",
];

/**
 * `--warm` の 2 本目以降が追記する user 発話（対話）— 性能相談の**続き**として自然な 12 問。
 *
 * MUST: 1 問ごとに新しい論点を 1 つだけ足す（前の答えの言い換えを頼まない）。言い換えを頼むと
 * 答えが前のターンの写しになり、{@link FREEFORM_WARM_FOLLOW_UPS} と同じ理由で受理率が跳ねる。
 * 先頭に置くのは直前の固定返答（buffer pool の展開）と語彙が重ならない問い — 1 本目の実測ターンで
 * 写しが起きると要約の中央値にそのまま入る。
 */
const DIALOGUE_WARM_FOLLOW_UPS: readonly string[] = [
  "Should the workgroup size be tuned per adapter, or is one value across vendors good enough?",
  "How do I keep shader translation from becoming the startup bottleneck on the first inference?",
  "What is a reasonable way to reason about occupancy on WebGPU when there is no profiler for it?",
  "Is there a safe way to overlap a compute pass with a buffer upload, or does the queue " +
  "serialise them anyway?",
  "Which limits should I request at device creation time so that no dispatch fails validation " +
  "mid-run?",
  "How should I size the buffer pool's size classes so that pooling does not just trade stalls " +
  "for wasted memory?",
  "How much does sharing one bind group layout across pipelines actually buy?",
  "When is a storage buffer the wrong choice, and a uniform buffer or a texture the right one?",
  "What is the cleanest way to test that two code paths produce bit-identical outputs on the GPU?",
  "How should I handle a lost device in the middle of a long generation?",
  "Does splitting one large matrix multiply into several dispatches ever help, or is it always " +
  "a loss?",
  "What should I record per dispatch so that a throughput regression is diagnosable afterwards?",
];

/**
 * `--warm` が使う追記の発話列（{@link buildWorkload} の会話の**続き**として流すもの）。
 *
 * 文書系（{@link DOCUMENT_WORKLOADS}）は発話列を持たない — 追記が文書そのものになって容量にも
 * 入らないが、落とす理由は「列が無い」ことなので、そう名指して落とす。
 */
export const warmFollowUps = (name: WorkloadName): readonly string[] => {
  if (isDocumentWorkload(name)) {
    throw new Error(
      `ワークロード ${name} は --warm 非対応（追記に流す user 発話の列を持たない — warm の対象: ` +
        `${WORKLOAD_NAMES.filter((one) => !isDocumentWorkload(one)).join(" / ")}）`,
    );
  }
  if (name === "dialogue") return DIALOGUE_WARM_FOLLOW_UPS;
  if (name === "freeform") return FREEFORM_WARM_FOLLOW_UPS;
  // 未知の綴りは落とす（`buildWorkload` と同じ扱い — 型を迂回した呼びを黙って通さない）。
  throw new Error(`未知のワークロード ${name}（既知: ${WORKLOAD_NAMES.join(" / ")}）`);
};

/**
 * `limit` 文字を超えない範囲で、**最後の段落境界**（空行）まで切り詰める（末尾は rstrip）。
 *
 * `export_drafter.py` の `truncate_paragraph` の鏡像。上限の手前に境界が無ければ落とす —
 * そこで文字数で切ると、生成の退化（同じ数語の繰り返し）が prompt 側の理由で起きる。
 */
export const truncateAtParagraph = (text: string, limit: number): string => {
  if (!Number.isInteger(limit) || limit < 1) {
    throw new Error(`文書の上限 ${limit} が 1 以上の整数でない`);
  }
  if (text.length <= limit) return text.trimEnd();
  // `limit - 2` から探すのは、境界の 2 文字が丸ごと上限の内側に入る位置だけを採るため
  // （Python 側の `rfind("\n\n", 0, limit)` と同じ範囲。JS の `lastIndexOf` は開始位置しか見ない）。
  const cut = text.lastIndexOf("\n\n", limit - 2);
  if (cut <= 0) throw new Error(`上限 ${limit} 文字の手前に段落境界（空行）が無い`);
  return text.slice(0, cut).trimEnd();
};

/** 素材の文書を読む（呼ばれたときだけ読む — モジュールスコープでは読まない）。 */
export const readWorkloadDocument = (): string => Deno.readTextFileSync(DOCUMENT_URL);

/** {@link buildWorkload} のノブ。 */
export type WorkloadOptions = {
  /**
   * 文書を切る文字数の上限（{@link DOCUMENT_WORKLOADS} のときだけ受ける）。
   *
   * 文書系以外に渡されたら落とす — 効かないノブを黙って受けると、出力に残る条件と実際に流した
   * prompt が食い違う。
   */
  readonly documentChars?: number;
};

/** ワークロード名 → 会話（system は付けない — 素の会話だけを測る）。 */
export const buildWorkload = (
  name: WorkloadName,
  options: WorkloadOptions = {},
): readonly Gemma4ChatMessage[] => {
  const documentWorkload = isDocumentWorkload(name);
  if (!documentWorkload && options.documentChars !== undefined) {
    throw new Error(
      `ワークロード ${name} は文書を読まないので documentChars を受けない` +
        `（文書系: ${DOCUMENT_WORKLOADS.join(" / ")}）`,
    );
  }
  if (documentWorkload) {
    const limit = options.documentChars;
    if (limit === undefined) throw new Error(`ワークロード ${name} は documentChars が必須`);
    const document = truncateAtParagraph(readWorkloadDocument(), limit);
    return [{
      role: "user",
      content: DOCUMENT_INSTRUCTION + document + DOCUMENT_QUESTION[name],
    }];
  }
  if (name === "dialogue") {
    return [
      { role: "user", content: DIALOGUE_QUESTION },
      { role: "assistant", content: DIALOGUE_ANSWER },
      { role: "user", content: DIALOGUE_FOLLOW_UP },
    ];
  }
  if (name === "freeform") return [{ role: "user", content: FREEFORM_PROMPT }];
  // 未知の綴りは落とす（CLI 側でも検査するが、型を迂回した呼びを黙って通さない）。
  throw new Error(`未知のワークロード ${name}（既知: ${WORKLOAD_NAMES.join(" / ")}）`);
};
