/**
 * The decode bench's status line: the visit (with its round and arm), then the pipeline's build or
 * the chat running (its prompt, and which of the prompt's chats; chat 0 is the cold one). The bench
 * reports a line before a build and before each chat's pause, so it is shown outside the measured
 * spans. Pure, so `vp test` covers it.
 */

import { armLabel, type BenchArm, benchPrompt } from "./plan.ts";
import type { BenchProgress } from "./run.ts";

/** The status line of `progress`, with the arms' labels; `chats` is the chats per prompt and visit. */
export const benchProgressLine = (
  progress: BenchProgress,
  arms: readonly BenchArm[],
  chats: number,
): string => {
  const { visit, visits } = progress;
  const where = `visit ${visit.visit + 1} / ${visits} (round ${visit.round + 1}) · arm ${visit.arm + 1} ${armLabel(arms[visit.arm])}`;
  return progress.kind === "loading"
    ? `${where} · building the pipeline`
    : `${where} · ${benchPrompt(progress.prompt).label} chat ${progress.chat + 1} / ${chats}${progress.chat === 0 ? " (cold)" : ""}`;
};
