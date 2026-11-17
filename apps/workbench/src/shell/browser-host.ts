/**
 * The browser's host of a run, shared by the entry panel and the batch panel: the environment from
 * an adapter of `navigator.gpu` (with the backend the person declared), the checkout of the dev
 * server (or the build), the distributions and the input files of the dev server, and the save through
 * `POST /workbench/results`.
 *
 * SPEC: docs/spec/workbench.md#the-shell
 */

import { readEnvironment } from "../environment.ts";
import type { RunHost } from "../execute.ts";
import type { WorkbenchRecord } from "../record.ts";
import { currentCheckout } from "./checkout.ts";
import { browserDistributions } from "./distributions.ts";
import { browserInputs } from "./inputs.ts";
import { type SaveResult, saveRecord } from "./save.ts";

/**
 * The browser's host. `declared` is the backend field's value when Run was clicked; `onEnded` is
 * told of each record once its run has ended, before the record is posted (the page says how the
 * run ended while the save is under way).
 */
export const browserHost = (
  declared: string,
  onEnded: (record: WorkbenchRecord) => void,
): RunHost<SaveResult> => ({
  readEnvironment: () =>
    readEnvironment(navigator.gpu, {
      host: "browser",
      userAgent: navigator.userAgent,
      crossOriginIsolated,
      ...(declared === "" ? {} : { backend: { name: declared, source: "declared" } }),
    }),
  readCheckout: currentCheckout,
  gpuOptions: {},
  openDistribution: browserDistributions(document.baseURI),
  openInput: browserInputs(document.baseURI),
  // A record the dev server does not take is "not saved" with a reason; the page carries on.
  save: (record, text) => {
    onEnded(record);
    return saveRecord(text);
  },
});
