/**
 * What the page shows after a run: the verdict (or the outcome of a run that did not finish), the
 * summary, where the record went, Download and Copy, and the record's text.
 *
 * SPEC: docs/spec/workbench.md#the-shell
 */

import { errorText, type WorkbenchRecord } from "@karume/sweep-app-support/record";

import { element } from "./dom.ts";
import { downloadRecord, type SaveResult } from "./save.ts";

/** The record text shown in the page; the download and the copy are always whole. */
const SHOWN_CHARACTERS = 200_000;

/** The status line once a run has made its record. */
export const endedText = (record: WorkbenchRecord): string =>
  record.outcome === "error"
    ? `Failed after ${record.seconds} s: ${record.error?.split("\n", 1)[0] ?? ""}`
    : `${record.outcome === "done" ? "Finished" : "Cancelled"} after ${record.seconds} s`;

export const resultView = (
  record: WorkbenchRecord,
  text: string,
  saved: SaveResult,
): HTMLElement => {
  const headline = record.verdict ?? (record.outcome === "done" ? undefined : record.outcome);
  const copy = element("button", { type: "button", textContent: "Copy record" });
  copy.addEventListener("click", () => {
    navigator.clipboard.writeText(text).then(
      () => {
        copy.textContent = "Copied";
      },
      (error: unknown) => {
        copy.textContent = `Copy failed: ${errorText(error)}`;
      },
    );
  });
  const download = element("button", { type: "button", textContent: "Download record" });
  download.addEventListener("click", () => downloadRecord(record, text));
  const left = text.length - SHOWN_CHARACTERS;
  return element("section", { className: "result" }, [
    ...(headline === undefined
      ? []
      : [element("p", { className: `verdict ${headline}`, textContent: headline.toUpperCase() })]),
    element("p", { textContent: `Outcome: ${record.outcome} · ${record.seconds} s` }),
    element(
      "ul",
      {},
      (record.summary ?? []).map((line) => element("li", { textContent: line })),
    ),
    element("p", {
      textContent:
        saved.kind === "saved"
          ? `Saved: ${saved.path}`
          : `Not saved: ${saved.reason}. Download the record to keep it.`,
    }),
    element("p", { className: "actions" }, [download, copy]),
    element("pre", { textContent: left > 0 ? text.slice(0, SHOWN_CHARACTERS) : text }),
    ...(left > 0
      ? [
          element("p", {
            textContent: `${left} characters left out (the download and the copy are whole).`,
          }),
        ]
      : []),
  ]);
};
