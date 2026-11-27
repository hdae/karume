/**
 * The selected batch's panel: its title, description and source; for a batch with problems, the
 * problems and no Run. Otherwise the table of its steps — a checkbox (all checked at first), the
 * number, the entry, the parameters the file wrote, a link that opens the step in the entry's form
 * and the step's state — then the backend field, Run / Cancel and the status line. A click on Run
 * runs the checked steps through {@link runBatch} on the browser's host; a finished step's row
 * opens the result view of a single run.
 *
 * SPEC: docs/spec/workbench.md#the-shell
 * SPEC: docs/spec/workbench.md#batches
 */

import type { GemvSweepSession } from "@karume/runtime/tune";
import { errorText, type WorkbenchRecord } from "@karume/sweep-app-support/record";

import {
  type ListedBatch,
  type ResolvedStep,
  TRACKED_BATCHES_DIRECTORY,
  UNTRACKED_BATCHES_DIRECTORY,
} from "../batch.ts";
import { batchSummary, runBatch, type StepState } from "../batch-run.ts";
import { browserHost } from "./browser-host.ts";
import { element } from "./dom.ts";
import { endedText, resultView } from "./result-view.ts";
import type { RunLock } from "./run-lock.ts";
import type { SaveResult } from "./save.ts";
import { batchQueryOf, queryOf } from "./url.ts";

const writeUrl = (id: string, backend: string): void => {
  history.replaceState(null, "", `?${batchQueryOf(id, backend).toString()}`);
};

const sourceText = (batch: ListedBatch): string =>
  batch.source === "tracked"
    ? `Tracked: ${TRACKED_BATCHES_DIRECTORY}/${batch.id}.json`
    : `Untracked (local): ${UNTRACKED_BATCHES_DIRECTORY}/${batch.id}.json`;

const firstLine = (text: string): string => text.split("\n", 1)[0] ?? "";

/** The outcome, the verdict when there is one, and the seconds (`done, verdict pass · 6.2 s`). */
const outcomeLine = (record: WorkbenchRecord): string =>
  `${record.outcome}${record.verdict === undefined ? "" : `, verdict ${record.verdict}`} · ${record.seconds} s`;

type StepRow = {
  readonly step: ResolvedStep;
  readonly checkbox: HTMLInputElement;
  readonly link: HTMLAnchorElement;
  readonly row: HTMLTableRowElement;
  readonly state: HTMLTableCellElement;
  /** The row under the step's row that holds its result view once the step ran. */
  readonly result: HTMLTableRowElement;
};

const COLUMNS = ["Include", "#", "Entry", "Parameters the file writes", "Form", "State"];

const stepRow = (step: ResolvedStep, lock: RunLock): StepRow => {
  const checkbox = element("input", {
    type: "checkbox",
    checked: true,
    ariaLabel: `Include step ${step.number}`,
  });
  const link = element("a", { textContent: "open in the form" });
  // Leaving the page would end the run; the menu is off while a run holds the lock, and so is this.
  link.addEventListener("click", (event) => {
    if (lock.running()) event.preventDefault();
  });
  const written = Object.entries(step.given);
  const state = element("td", { className: "state" });
  const row = element("tr", {}, [
    element("td", {}, [checkbox]),
    element("td", { textContent: String(step.number) }),
    element("td", {}, [element("code", { textContent: step.entry.id })]),
    element(
      "td",
      {},
      written.length === 0
        ? [element("span", { className: "defaults", textContent: "defaults" })]
        : written.map(([name, value]) =>
            element("div", {}, [element("code", { textContent: `${name}=${value}` })]),
          ),
    ),
    element("td", {}, [link]),
    state,
  ]);
  const result = element("tr", { className: "step-result", hidden: true }, [
    element("td", { colSpan: COLUMNS.length }),
  ]);
  return { step, checkbox, link, row, state, result };
};

const showState = (row: StepRow, state: StepState<SaveResult>): void => {
  row.row.classList.remove("running");
  const cell = row.result.cells[0];
  switch (state.kind) {
    case "not-selected":
      row.state.replaceChildren("not selected");
      return;
    case "not-run":
      row.state.replaceChildren("did not run");
      return;
    case "host-failed":
      row.state.replaceChildren(
        element("div", {
          className: "error",
          textContent: `failed around the run: ${firstLine(errorText(state.error))}`,
        }),
        element("div", { textContent: "The batch stopped here." }),
      );
      return;
    case "ran": {
      const { record, text, saved } = state.run;
      const headline = record.verdict ?? record.outcome;
      const first =
        record.outcome === "error" ? firstLine(record.error ?? "") : record.summary?.[0];
      row.state.replaceChildren(
        element("div", { className: headline, textContent: outcomeLine(record) }),
        ...(first === undefined || first === "" ? [] : [element("div", { textContent: first })]),
        element("div", {
          textContent:
            saved.kind === "saved" ? `Saved: ${saved.path}` : `Not saved: ${saved.reason}`,
        }),
      );
      cell?.replaceChildren(
        element("details", {}, [
          element("summary", { textContent: `Result of step ${row.step.number}` }),
          resultView(record, text, saved),
        ]),
      );
      row.result.hidden = false;
      return;
    }
  }
};

export const batchPanel = (options: {
  readonly batch: ListedBatch;
  readonly lock: RunLock;
  /** The backend field, shared by every selection (its value carries over). */
  readonly backend: HTMLInputElement;
  /** The page load's session, which a sweep's record names. */
  readonly session: GemvSweepSession;
  /** `undefined` when Run can be offered, else what to say in its place. */
  readonly gpuProblem: Promise<string | undefined>;
  /** Aborted when another entry or batch is selected, which removes this panel's listener on `backend`. */
  readonly selection: AbortSignal;
}): HTMLElement => {
  const { batch, lock, backend } = options;
  writeUrl(batch.id, backend.value.trim());
  const heading = [
    element("h2", { textContent: batch.title }),
    ...(batch.description === undefined ? [] : [element("p", { textContent: batch.description })]),
    element("p", { className: "source", textContent: sourceText(batch) }),
  ];
  if (batch.kind === "problems") {
    return element("section", { className: "panel batch" }, [
      ...heading,
      element("p", { textContent: "This batch has problems, so it does not run:" }),
      element(
        "ul",
        { className: "problems" },
        batch.problems.map((problem) => element("li", { textContent: problem })),
      ),
    ]);
  }

  const rows = batch.steps.map((step) => stepRow(step, lock));
  const linkRows = (): void => {
    for (const { step, link } of rows) {
      link.href = `?${queryOf(step.entry, step.parameters, backend.value.trim()).toString()}`;
    }
  };
  linkRows();
  backend.addEventListener(
    "change",
    () => {
      writeUrl(batch.id, backend.value.trim());
      linkRows();
    },
    { signal: options.selection },
  );

  const status = element("p", { className: "status" });
  const actions = element("p", {
    className: "actions",
    textContent: "Looking for a WebGPU adapter…",
  });
  const run = element("button", { type: "button", textContent: "Run" });
  const cancel = element("button", { type: "button", textContent: "Cancel", disabled: true });
  let controller: AbortController | undefined;
  cancel.addEventListener("click", () => controller?.abort());
  run.addEventListener("click", () => {
    if (lock.running()) return;
    const selected = rows.filter(({ checkbox }) => checkbox.checked).map(({ step }) => step.number);
    if (selected.length === 0) {
      status.textContent = "Check at least one step.";
      return;
    }
    controller = new AbortController();
    const { signal } = controller;
    lock.take({ run, cancel });
    for (const row of rows) {
      row.checkbox.disabled = true;
      row.row.classList.remove("running");
      row.result.hidden = true;
      row.result.cells[0]?.replaceChildren();
      row.state.replaceChildren(row.checkbox.checked ? "waiting" : "not selected");
    }
    status.textContent = "Starting…";
    const at = (step: ResolvedStep): string => `step ${step.number} of ${batch.steps.length}`;
    let current: ResolvedStep | undefined;
    const host = browserHost(
      backend.value.trim(),
      (record) => {
        if (current !== undefined) status.textContent = `${at(current)} · ${endedText(record)}`;
      },
      options.session,
    );
    void (async () => {
      try {
        const ran = await runBatch({
          batch,
          selected,
          signal,
          host,
          onEvent: (event) => {
            const row = rows[event.step.number - 1];
            switch (event.kind) {
              case "started":
                current = event.step;
                row?.row.classList.add("running");
                row?.state.replaceChildren("running…");
                status.textContent = `${at(event.step)} · Starting…`;
                return;
              case "progress":
                status.textContent = `${at(event.step)} · ${event.text}`;
                return;
              case "settled":
                if (row !== undefined) showState(row, event.state);
                return;
            }
          },
        });
        const summary = batchSummary(batch.id, ran);
        const stop = ran.steps.flatMap(({ step, state }) =>
          state.kind === "host-failed" ? [{ step, error: state.error }] : [],
        )[0];
        status.textContent =
          stop === undefined
            ? summary
            : `${at(stop.step)} · The batch stopped: ${firstLine(errorText(stop.error))} — ${summary}`;
      } catch (error) {
        status.textContent = `The batch could not run: ${errorText(error)}`;
      } finally {
        lock.release();
        for (const row of rows) row.checkbox.disabled = false;
      }
    })();
  });

  void options.gpuProblem.then((problem) => {
    if (problem === undefined) actions.replaceChildren(run, " ", cancel);
    else actions.textContent = problem;
  });

  return element("section", { className: "panel batch" }, [
    ...heading,
    element("table", { className: "steps" }, [
      element("thead", {}, [
        element(
          "tr",
          {},
          COLUMNS.map((column) => element("th", { textContent: column })),
        ),
      ]),
      element(
        "tbody",
        {},
        rows.flatMap(({ row, result }) => [row, result]),
      ),
    ]),
    element("label", { className: "backend" }, ["backend (optional) ", backend]),
    actions,
    status,
  ]);
};
