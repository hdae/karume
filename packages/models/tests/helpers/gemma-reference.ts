/**
 * The **sha256 reference row of one quant seat** of the gemma lines — the regular `karume-gemma4`,
 * the q4_0 QAT `karume-gemma4-qat` and the mobile QAT `karume-gemma4-qat-mobile`: how the run and
 * its artifact are made, and the one path to the settlement (shared by
 * `e2e_gemma4_reference_test.ts` / `e2e_gemma4_qat_reference_test.ts` /
 * `e2e_gemma4_qat_mobile_reference_test.ts`).
 *
 * ## What is pinned
 *
 * The **token id sequence** of one fixed prompt ({@link REFERENCE_PROMPT}) run greedily up to
 * {@link REFERENCE_MAX_NEW_TOKENS} tokens. An LLM's output is a discrete id sequence, so, like the
 * sha checks of PNG / WAV, the output artifact itself is pinned (ADR 0106). The claim is only "not
 * a single token moved on the same machine with the same preset"; it says nothing about
 * cross-device identity or quality.
 *
 * ## Row classes (reference rows / practical rows)
 *
 * A row's class is **derived from the preset's `session`** (ADR 0110 decision 7): a preset with an
 * empty `session` is a reference row (a freeze of the reference layer — a change is a regression),
 * a non-empty one is a practical row (a regression + determinism detector — a commit that changes
 * the practical layer runs `KARUME_REFERENCE=rewrite` in the same commit). Callers do not keep the
 * presets in a fixed table; {@link mirrorSeats} enumerates **every quant preset** from the mirror's
 * manifest — the row class is derived there too, so the derivation itself is the source of truth
 * (a preset added to the distribution adds a row, and a change in a preset's `session` changes its
 * class with it. A fixed table would silently leave a hole where a distributed preset has no row).
 *
 * ## The artifact
 *
 * A JSON text `{ "ids": [...] }` (2-space indent, trailing newline — {@link seatArtifactBytes}).
 * The sha is taken over these bytes. The decoded text and the stop reason go only into the `note`
 * of `results.json` — the text is derived from the id sequence (the same change is not counted
 * twice), so it is not put in the artifact.
 */

import { readTextFileSync } from "@karume/test-support/fs";
import { assert } from "@std/assert";
import type { ChatMessage, ChatOptions, ChatStop, ChatStream } from "../../gemma.ts";
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
 * The fixed prompt (one English text). It asks for a response longer than
 * {@link REFERENCE_MAX_NEW_TOKENS} — with a prompt that closes early with EOS, the pinned id
 * sequence would be shorter and the check weaker.
 */
export const REFERENCE_PROMPT = "Explain in about five sentences why the sky is blue.";

/**
 * The upper bound on sampled tokens (the stop token counts toward it — `ChatOptions.maxNewTokens`).
 */
export const REFERENCE_MAX_NEW_TOKENS = 64;

/** The pipeline surface a run needs (common to `gemma.Pipeline` and `gemma4QatMobile.Pipeline`). */
export type ChatPipeline = {
  chat(messages: readonly ChatMessage[], options: ChatOptions): ChatStream;
  dispose(): Promise<void>;
};

/** A row's class ("Row classes" in the module doc). */
export type SeatRow = "reference" | "practical";

/** One quant preset (model × quant) and its row class (derived from the preset's `session`). */
export type Seat<Model extends string = string> = {
  readonly model: Model;
  readonly quant: string;
  readonly row: SeatRow;
};

/** The case ID of a reference value (`<model>/<quant>`). */
export const seatCaseId = (seat: Pick<Seat, "model" | "quant">): string =>
  `${seat.model}/${seat.quant}`;

/**
 * The artifact's file name (relative to the results directory — `/` cannot appear, so the parts are
 * joined with `-`).
 */
export const seatArtifactName = (seat: Pick<Seat, "model" | "quant">): string =>
  `${seat.model}-${seat.quant}.json`;

/**
 * Enumerates **every model × every quant preset** from the mirror's manifest and derives each row's
 * class from whether the preset's `session` is empty ("Row classes" in the module doc). Called when
 * tests are registered, so it reads synchronously.
 *
 * An empty set when the mirror is missing or its distribution cannot be read by this version
 * (`mirrorAvailable` is false) — callers SKIP and give the reason. The order is the manifest's
 * declaration order (model → quant).
 */
export const mirrorSeats = (mirror: URL): readonly Seat[] => {
  if (!mirrorAvailable(mirror)) return [];
  const manifest = parseManifest(readTextFileSync(new URL("karume.json", mirror)));
  return Object.entries(manifest.models).flatMap(([model, entry]) =>
    Object.entries(entry.quants).map(([quant, { session }]): Seat => ({
      model,
      quant,
      row: Object.keys(session).length === 0 ? "reference" : "practical",
    })),
  );
};

/** The result of running one preset. */
export type SeatRun = {
  /** The generated token ids (without the stop token — in the order `onToken` received them). */
  readonly ids: readonly number[];
  readonly stop: ChatStop;
  /** The decoded text (only put in the `note` — not in the artifact). */
  readonly text: string;
};

/**
 * Runs one preset: open → run the fixed prompt once, greedily → release.
 *
 * `speculative: false` keeps the row to **the preset's own numerics** (with a drafter, tokens that
 * went through the speculative verify form would be mixed in. Identity with the drafter is checked
 * separately by `e2e_gemma4_speculative_test.ts`). `false` also passes on a pipeline without a
 * drafter (an explicit opt-out does not require a drafter).
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

/**
 * The artifact's bytes (`{ "ids": [...] }`, 2-space indent, trailing newline — the sha is taken
 * over these bytes).
 */
export const seatArtifactBytes = (ids: readonly number[]): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(`${JSON.stringify({ ids }, undefined, 2)}\n`);

/** The `note` of `results.json` (stop reason, token count, text). */
const seatNote = (run: SeatRun): string =>
  `stop ${run.stop.reason} / ${run.stop.tokens} tokens / text ${JSON.stringify(run.text)}`;

/**
 * The body of one preset's sha check: adapter identity → run → the precondition of the check's
 * strength (it generated up to the limit) → write the artifact and compare → record the settlement
 * → fail **after recording** (throwing first would leave the measured sha out of the results).
 *
 * The whole is wrapped in `runRecordedCase` — wherever it throws, a `fail` settlement stays in the
 * results (leaving without a settlement would leave the settlement of an earlier run of the same
 * day in place).
 *
 * Callers register this case with `ignore: !RUNNABLE || references.lacksReference(id)` (in compare
 * mode a case without a row never gets here — `References.check` throws).
 */
export const settleSeatReference = async <Model extends string>(context: {
  readonly references: References;
  readonly results: Results;
  readonly seat: Seat<Model>;
  readonly open: (choice: {
    readonly model: Model;
    readonly quant: string;
  }) => Promise<ChatPipeline>;
}): Promise<void> => {
  const { references, results, seat, open } = context;
  const id = seatCaseId(seat);
  /** The sha settlement (taken outside so that a mismatch fails after recording). */
  let settlement: ReferenceSettlement | undefined;
  await runRecordedCase(
    results,
    {
      id,
      failureNote: (cause) => (cause instanceof Error ? cause.message : String(cause)),
    },
    async () => {
      // This path can write reference values, so the identity of the adapter the key was taken on
      // and the running adapter is checked first (the pipeline calls acquireGpu internally, so we
      // do not hold a GpuContext ourselves).
      await assertRunningAdapter();
      const run = await runSeat(open, seat);
      // The precondition of the check's strength (it generated up to the limit = the pinned id
      // sequence has {@link REFERENCE_MAX_NEW_TOKENS} ids). Checked before settling — no row is
      // made from a run that closed early with EOS. The text goes into the diagnostic (it becomes
      // the failure's note).
      assert(
        run.stop.reason === "max-tokens",
        `${id}: the response closed after ${run.ids.length} tokens (${seatNote(run)}) — ` +
          `the fixed prompt is expected to produce a response longer than ${REFERENCE_MAX_NEW_TOKENS} tokens`,
      );
      const artifact = seatArtifactName(seat);
      settlement = await settleReference(references, results, {
        id,
        artifact,
        bytes: seatArtifactBytes(run.ids),
      });
      return { ...referenceEntryFields(settlement, artifact), note: seatNote(run) };
    },
  );
  if (settlement?.check.status === "fail") {
    throw new Error(referenceMismatchMessage(id, settlement, references));
  }
};
