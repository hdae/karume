/**
 * Where a browser run's record goes: under the dev server it is POSTed to `/workbench/results`,
 * which writes it under `outputs/workbench/`; a production build saves nothing (the page offers the
 * download). The download is the same text either way.
 */

import { utcStamp, type WorkbenchRecord } from "../record.ts";

export type SaveResult =
  | { readonly kind: "saved"; readonly path: string }
  | { readonly kind: "not-saved"; readonly reason: string };

export const saveRecord = async (text: string): Promise<SaveResult> => {
  if (!import.meta.env.DEV) {
    return { kind: "not-saved", reason: "a production build does not write records" };
  }
  let response: Response;
  try {
    response = await fetch("/workbench/results", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: text,
    });
  } catch (error) {
    return { kind: "not-saved", reason: `the dev server could not be reached (${String(error)})` };
  }
  if (response.status !== 201) {
    const reason = (await response.text()).trim();
    return { kind: "not-saved", reason: `the dev server answered ${response.status}: ${reason}` };
  }
  const body: unknown = await response.json();
  return typeof body === "object" &&
    body !== null &&
    "path" in body &&
    typeof body.path === "string"
    ? { kind: "saved", path: body.path }
    : { kind: "not-saved", reason: "the dev server answered no path" };
};

/** Hands the record text to the browser as `<entry id>_<UTC stamp>.json`. */
export const downloadRecord = (record: WorkbenchRecord, text: string): void => {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${record.entry.id}_${utcStamp(new Date(record.recordedAt))}.json`;
  anchor.click();
  // The click starts the download synchronously; revoke on the next task.
  setTimeout(() => URL.revokeObjectURL(url), 0);
};
