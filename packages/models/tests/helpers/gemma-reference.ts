/**
 * gemma 系列（gemma4 / gemma4-qat）の **quant 席 1 つぶんの sha 参照行** — 実走と実物の作り方と、
 * 決着までの 1 本（`e2e_gemma4_reference_test.ts` / `e2e_gemma4_qat_reference_test.ts` が共有する）。
 *
 * ## 何を固定するのか
 *
 * 固定プロンプト 1 本（{@link REFERENCE_PROMPT}）を greedy で {@link REFERENCE_MAX_NEW_TOKENS}
 * token まで回した **token id 列**。LLM の出口は離散の id 列なので、PNG / WAV の sha 門と同じく
 * 出口の成果物そのものを固定する（ADR 0106）。主張は「同じ機・同じ席で 1 token も動いていない」
 * だけで、クロスデバイスの同一性も品質も言わない。
 *
 * ## 行のクラス（参照行 / 実用行）
 *
 * 行のクラスは**席の `session` から導かれる**（ADR 0110 決定 7）: `session` が空の席は参照行
 * （参照層の凍結 — 変わったら退行）、非空の席は実用行（退行 + 決定性の検出器 — 実用層を変えた
 * コミットでは同じコミットで `KARUME_REFERENCE=rewrite` する）。呼び手は席を固定表で持たず、
 * {@link mirrorSeats} でミラーの manifest から**全 quant 席**を列挙する — 行のクラスもそこで
 * 導くので、導出そのものが正本になる（配布形に席が増えれば行も増え、席の `session` が変われば
 * クラスも追従する。固定表だと、配布済みの席に行が無い穴が黙って残る）。
 *
 * ## 実物
 *
 * `{ "ids": [...] }` の JSON テキスト（2 スペース・末尾改行 — {@link seatArtifactBytes}）。sha は
 * このバイト列に対して採る。復号後の本文と停止理由は `results.json` の `note` にだけ載せる —
 * 本文は id 列から導かれる（同じ変化を 2 度数えない）ので、実物には入れない。
 */

import { assert } from "@std/assert";
import type {
  Gemma4ChatMessage,
  Gemma4ChatOptions,
  Gemma4ChatStop,
  Gemma4ChatStream,
} from "../../gemma.ts";
import { parseManifest } from "@karume/hub";
import { assertRunningAdapter } from "../../../runtime/tests/helpers/environment.ts";
import {
  referenceEntryFields,
  referenceMismatchMessage,
  type References,
  type ReferenceSettlement,
  settleReference,
} from "../../../runtime/tests/helpers/reference.ts";
import { type Results, runRecordedCase } from "../../../runtime/tests/helpers/results.ts";
import { mirrorAvailable } from "./gemma-mirror.ts";

/**
 * 固定プロンプト（英文 1 本）。{@link REFERENCE_MAX_NEW_TOKENS} を超える応答が出る依頼にしてある —
 * 早く EOS で閉じるプロンプトだと、固定される id 列が短くなって門が弱くなる。
 */
export const REFERENCE_PROMPT = "Explain in about five sentences why the sky is blue.";

/** 生成する token 数の上限（停止 token はこの数に含まれない — `Gemma4ChatOptions.maxNewTokens`）。 */
export const REFERENCE_MAX_NEW_TOKENS = 64;

/** 走行に要る pipeline の面（`Gemma4Pipeline` / `Gemma4QatPipeline` の共通部分）。 */
export type ChatPipeline = {
  chat(messages: readonly Gemma4ChatMessage[], options: Gemma4ChatOptions): Gemma4ChatStream;
  dispose(): Promise<void>;
};

/** 行のクラス（モジュール doc の「行のクラス」）。 */
export type SeatRow = "reference" | "practical";

/** quant 席 1 つ（model × quant）と、その行のクラス（席の `session` から導いたもの）。 */
export type Seat<Model extends string = string> = {
  readonly model: Model;
  readonly quant: string;
  readonly row: SeatRow;
};

/** 参照値のケース ID（`<model>/<quant>`）。 */
export const seatCaseId = (seat: Pick<Seat, "model" | "quant">): string =>
  `${seat.model}/${seat.quant}`;

/** 実物のファイル名（結果の席からの相対 — `/` を含められないので `-` で繋ぐ）。 */
export const seatArtifactName = (seat: Pick<Seat, "model" | "quant">): string =>
  `${seat.model}-${seat.quant}.json`;

/**
 * ミラーの manifest から**全 model × 全 quant 席**を列挙し、行のクラスを席の `session` の空非空
 * から導く（モジュール doc「行のクラス」）。テストの登録時に呼ぶので同期で読む。
 *
 * ミラーが無い・この版が読めない配布形（`mirrorAvailable` が偽）なら空集合 — 呼び手は理由を
 * 名乗って SKIP する。並びは manifest の宣言順（model → quant）。
 */
export const mirrorSeats = (mirror: URL): readonly Seat[] => {
  if (!mirrorAvailable(mirror)) return [];
  const manifest = parseManifest(Deno.readTextFileSync(new URL("karume.json", mirror)));
  return Object.entries(manifest.models).flatMap(([model, entry]) =>
    Object.entries(entry.quants).map(([quant, { session }]): Seat => ({
      model,
      quant,
      row: Object.keys(session).length === 0 ? "reference" : "practical",
    }))
  );
};

/** 1 席ぶんの走行の結果。 */
export type SeatRun = {
  /** 生成 token の id 列（停止 token は含まない — `onToken` が受けた順）。 */
  readonly ids: readonly number[];
  readonly stop: Gemma4ChatStop;
  /** 復号後の本文（`note` に載せるだけ — 実物には入れない）。 */
  readonly text: string;
};

/**
 * 1 席ぶんを走らせる: 開く → 固定プロンプト 1 本を greedy で回す → 解放する。
 *
 * `speculative: false` は行を**席の数値だけ**にするため（drafter が居ると投機の verify 形を
 * 通る token が混ざる。drafter との同一性は `e2e_gemma4_speculative_test.ts` が別に門にしている）。
 * drafter を持たない pipeline でも `false` は通る（明示の取り消しは drafter を要求しない）。
 */
export const runSeat = async <Model extends string>(
  open: (choice: { readonly model: Model; readonly quant: string }) => Promise<ChatPipeline>,
  seat: Seat<Model>,
): Promise<SeatRun> => {
  const pipeline = await open({ model: seat.model, quant: seat.quant });
  try {
    const ids: number[] = [];
    const stream = pipeline.chat([{ role: "user", content: REFERENCE_PROMPT }], {
      maxNewTokens: REFERENCE_MAX_NEW_TOKENS,
      sampler: { temperature: 0 },
      speculative: false,
      onToken: (id) => ids.push(id),
    });
    const text = await stream.text();
    const stop = await stream.done;
    return { ids, stop, text };
  } finally {
    await pipeline.dispose();
  }
};

/** 実物のバイト列（`{ "ids": [...] }`・2 スペース・末尾改行 — sha はこのバイト列に対して採る）。 */
export const seatArtifactBytes = (ids: readonly number[]): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(`${JSON.stringify({ ids }, undefined, 2)}\n`);

/** `results.json` の `note`（停止理由・token 数・本文）。 */
const seatNote = (run: SeatRun): string =>
  `stop ${run.stop.reason} / ${run.stop.tokens} tokens / text ${JSON.stringify(run.text)}`;

/**
 * 1 席の sha 門の本体: アダプタの同一性 → 走行 → 門の強さの前提（上限まで生成した）→ 実物を
 * 書いて突き合わせる → 決着を積む → **積んだ後に**落とす（先に投げると実測 sha が結果に残らない）。
 *
 * 全体を `runRecordedCase` で包む — どこで投げても `fail` の決着が席に残る（決着の無いまま
 * 抜けると、同じ日の前回の走行の決着が居座る）。
 *
 * 呼び手はこのケースを `ignore: !RUNNABLE || references.lacksReference(id)` で登録する
 * （比較モードで行が無いケースはここまで来ない — `References.check` が投げる）。
 */
export const settleSeatReference = async <Model extends string>(context: {
  readonly references: References;
  readonly results: Results;
  readonly seat: Seat<Model>;
  readonly open: (
    choice: { readonly model: Model; readonly quant: string },
  ) => Promise<ChatPipeline>;
}): Promise<void> => {
  const { references, results, seat, open } = context;
  const id = seatCaseId(seat);
  /** sha の決着（記録の後で不一致を落とすために外へ持ち出す）。 */
  let settlement: ReferenceSettlement | undefined;
  await runRecordedCase(results, {
    id,
    failureNote: (cause) => cause instanceof Error ? cause.message : String(cause),
  }, async () => {
    // 参照値を書きうる経路なので、キーを採ったアダプタと実行アダプタの同一性を先に見る
    // （pipeline は内部で acquireGpu するので、自分では GpuContext を持たない）。
    await assertRunningAdapter();
    const run = await runSeat(open, seat);
    // 門の強さの前提（上限まで生成した = 固定した id 列が {@link REFERENCE_MAX_NEW_TOKENS} 本ある）。
    // settle より前に見る — 早く EOS で閉じた走行から行を作らない。本文は診断文に載る（失敗の
    // note になる）。
    assert(
      run.stop.reason === "max-tokens",
      `${id}: 応答が ${run.ids.length} token で閉じた（${seatNote(run)}）— ` +
        `固定プロンプトは ${REFERENCE_MAX_NEW_TOKENS} token を超える応答を出す前提`,
    );
    const artifact = seatArtifactName(seat);
    settlement = await settleReference(references, results, {
      id,
      artifact,
      bytes: seatArtifactBytes(run.ids),
    });
    return { ...referenceEntryFields(settlement, artifact), note: seatNote(run) };
  });
  if (settlement?.check.status === "fail") {
    throw new Error(referenceMismatchMessage(id, settlement, references));
  }
};
