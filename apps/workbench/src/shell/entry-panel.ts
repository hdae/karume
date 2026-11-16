/**
 * The selected entry's panel: its parameter form, the backend field, Run / Cancel, the status line
 * and, after a run, the result. The form and the backend field keep the URL in step; a click on Run
 * runs the entry through {@link runEntry} on the browser's host and shows what it made.
 *
 * SPEC: docs/spec/workbench.md#the-shell
 */

import { type Entry, parseParameters, type ParameterValues } from "../entry.ts";
import { runEntry } from "../execute.ts";
import { errorText } from "../record.ts";
import { browserHost } from "./browser-host.ts";
import { element } from "./dom.ts";
import { parameterForm } from "./form.ts";
import { endedText, resultView } from "./result-view.ts";
import type { RunLock } from "./run-lock.ts";
import { queryOf } from "./url.ts";

const writeUrl = (entry: Entry, values: ParameterValues, backend: string): void => {
  history.replaceState(null, "", `?${queryOf(entry, values, backend).toString()}`);
};

export const entryPanel = (options: {
  readonly entry: Entry;
  /** The parameter values as strings (the URL's); values that do not parse fall back to the defaults. */
  readonly given: Readonly<Record<string, string>>;
  readonly lock: RunLock;
  /** The backend field, shared by every selection (its value carries over). */
  readonly backend: HTMLInputElement;
  /** `undefined` when Run can be offered, else what to say in its place. */
  readonly gpuProblem: Promise<string | undefined>;
  /** Aborted when another entry is selected, which removes this panel's listener on `backend`. */
  readonly selection: AbortSignal;
}): HTMLElement => {
  const { entry, lock, backend } = options;
  const status = element("p", { className: "status" });
  let values: ParameterValues;
  try {
    values = parseParameters(entry.parameters, options.given);
  } catch (error) {
    status.textContent = `${errorText(error)} — the defaults are shown instead.`;
    values = parseParameters(entry.parameters, {});
  }
  const form = parameterForm(entry.parameters, values);
  const actions = element("p", {
    className: "actions",
    textContent: "Looking for a WebGPU adapter…",
  });
  const result = element("div");
  const sync = (): void => {
    try {
      writeUrl(entry, parseParameters(entry.parameters, form.read()), backend.value.trim());
    } catch (error) {
      status.textContent = errorText(error);
    }
  };
  form.element.addEventListener("change", sync);
  backend.addEventListener("change", sync, { signal: options.selection });
  writeUrl(entry, values, backend.value.trim());

  const run = element("button", { type: "button", textContent: "Run" });
  const cancel = element("button", { type: "button", textContent: "Cancel", disabled: true });
  let controller: AbortController | undefined;
  cancel.addEventListener("click", () => controller?.abort());
  run.addEventListener("click", () => {
    if (lock.running()) return;
    let parameters: ParameterValues;
    try {
      parameters = parseParameters(entry.parameters, form.read());
    } catch (error) {
      status.textContent = errorText(error);
      return;
    }
    controller = new AbortController();
    const { signal } = controller;
    lock.take({ run, cancel });
    result.replaceChildren();
    status.textContent = "Starting…";
    const host = browserHost(backend.value.trim(), (record) => {
      status.textContent = endedText(record);
    });
    void (async () => {
      try {
        const { record, text, saved } = await runEntry(host, {
          entry,
          parameters,
          signal,
          progress: (line) => {
            status.textContent = line;
          },
        });
        result.replaceChildren(resultView(record, text, saved));
      } catch (error) {
        status.textContent = `The run could not start: ${errorText(error)}`;
      } finally {
        lock.release();
      }
    })();
  });

  void options.gpuProblem.then((problem) => {
    if (problem === undefined) actions.replaceChildren(run, " ", cancel);
    else actions.textContent = problem;
  });

  return element("section", { className: "panel" }, [
    element("h2", { textContent: entry.title }),
    element("p", { textContent: entry.description }),
    form.element,
    element("label", { className: "backend" }, ["backend (optional) ", backend]),
    actions,
    status,
    result,
  ]);
};
