/**
 * The shell: the list of entries, then the list of batches (in their own colour), the selected
 * entry's panel (`shell/entry-panel.ts`) or batch's panel (`shell/batch-panel.ts`) and the first
 * selection from the URL. The selected entry, its parameters or the selected batch, and the
 * declared backend live in the URL, so a link reproduces a run setup; nothing starts without a click
 * on Run. One run at a time (`shell/run-lock.ts`).
 *
 * SPEC: docs/spec/workbench.md#the-shell
 */

import "./style.css";

import { type BatchFile, type ListedBatch, listBatches } from "./batch.ts";
import { trackedBatchFiles } from "./batches.ts";
import { registeredEntries } from "./entries.ts";
import type { Entry } from "./entry.ts";
import { errorText } from "./record.ts";
import { batchPanel } from "./shell/batch-panel.ts";
import { element } from "./shell/dom.ts";
import { entryPanel } from "./shell/entry-panel.ts";
import { createRunLock } from "./shell/run-lock.ts";
import { fetchUntrackedBatches } from "./shell/untracked-batches.ts";
import { BACKEND_KEY, givenOfQuery, selectionOfQuery } from "./shell/url.ts";

/** `undefined` when Run can be offered, else what to say in its place. */
const probeGpu = async (): Promise<string | undefined> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return "This browser has no WebGPU (navigator.gpu is missing).";
  try {
    return (await gpu.requestAdapter()) === null
      ? "WebGPU gives no adapter in this browser."
      : undefined;
  } catch (error) {
    return `WebGPU failed to give an adapter: ${errorText(error)}`;
  }
};

/** The untracked batch files, or why the dev server could not list them. */
const untrackedOrProblem = async (): Promise<readonly BatchFile[] | string> => {
  try {
    return await fetchUntrackedBatches();
  } catch (error) {
    return `The untracked batches could not be listed: ${errorText(error)}`;
  }
};

const batchButton = (batch: ListedBatch): HTMLButtonElement =>
  element("button", { type: "button" }, [
    element("strong", { textContent: batch.title }),
    ...(batch.description === undefined
      ? []
      : [element("span", { textContent: batch.description })]),
    ...(batch.source === "untracked" || batch.kind === "problems"
      ? [
          element("span", { className: "marks" }, [
            ...(batch.source === "untracked"
              ? [element("small", { className: "mark", textContent: "local" })]
              : []),
            ...(batch.kind === "problems"
              ? [element("small", { className: "mark problems", textContent: "has problems" })]
              : []),
          ]),
        ]
      : []),
  ]);

const start = async (): Promise<void> => {
  const root = document.getElementById("app");
  if (root === null) throw new Error("index.html has no #app");
  const entries = registeredEntries();
  const untracked = await untrackedOrProblem();
  const batches = listBatches({
    tracked: trackedBatchFiles(),
    untracked: typeof untracked === "string" ? [] : untracked,
    entries,
    production: !import.meta.env.DEV,
  });
  const query = new URLSearchParams(location.search);
  const backend = element("input", {
    type: "text",
    value: query.get(BACKEND_KEY) ?? "",
    placeholder: "d3d12, metal, vulkan…",
  });
  const entryButtons = new Map<Entry, HTMLButtonElement>(
    entries.map((entry): [Entry, HTMLButtonElement] => [
      entry,
      element("button", { type: "button" }, [
        element("strong", { textContent: entry.title }),
        element("span", { textContent: entry.description }),
      ]),
    ]),
  );
  const batchButtons = new Map<ListedBatch, HTMLButtonElement>(
    batches.map((batch): [ListedBatch, HTMLButtonElement] => [batch, batchButton(batch)]),
  );
  const buttons = [...entryButtons.values(), ...batchButtons.values()];
  const lock = createRunLock(buttons);
  const nav = element("nav", {}, [
    element(
      "ul",
      { className: "entries" },
      [...entryButtons.values()].map((button) => element("li", {}, [button])),
    ),
    element(
      "ul",
      { className: "entries batches" },
      [...batchButtons.values()].map((button) => element("li", {}, [button])),
    ),
    ...(typeof untracked === "string"
      ? [element("p", { className: "status", textContent: untracked })]
      : []),
  ]);
  const layout = element("div", { className: "layout" });
  // Asked on the first selection, once for the page.
  let gpuProblem: Promise<string | undefined> | undefined;
  let selection: AbortController | undefined;

  /** Marks `current` in the menu and puts the panel `panelOf` builds next to it. */
  const show = (
    current: HTMLButtonElement | undefined,
    panelOf: (selected: AbortSignal, gpu: Promise<string | undefined>) => HTMLElement,
  ): void => {
    for (const button of buttons) button.setAttribute("aria-current", String(button === current));
    selection?.abort();
    selection = new AbortController();
    gpuProblem ??= probeGpu();
    layout.replaceChildren(nav, panelOf(selection.signal, gpuProblem));
  };
  const selectEntry = (entry: Entry, given: Readonly<Record<string, string>>): void =>
    show(entryButtons.get(entry), (selected, gpu) =>
      entryPanel({ entry, given, lock, backend, gpuProblem: gpu, selection: selected }),
    );
  const selectBatch = (batch: ListedBatch): void =>
    show(batchButtons.get(batch), (selected, gpu) =>
      batchPanel({ batch, lock, backend, gpuProblem: gpu, selection: selected }),
    );
  for (const [entry, button] of entryButtons) {
    button.addEventListener("click", () => selectEntry(entry, {}));
  }
  for (const [batch, button] of batchButtons) {
    button.addEventListener("click", () => selectBatch(batch));
  }

  root.replaceChildren(
    element("header", {}, [
      element("h1", { textContent: "Karume workbench" }),
      element("p", {
        textContent: import.meta.env.DEV
          ? "Records are saved under outputs/workbench/ by the dev server."
          : "A production build saves no records: download them.",
      }),
    ]),
    layout,
  );

  const notice = (text: string): void => {
    layout.replaceChildren(nav, element("section", { className: "panel", textContent: text }));
  };
  const wanted = selectionOfQuery(query);
  switch (wanted.kind) {
    case "none":
      notice("Pick an entry or a batch.");
      return;
    case "both":
      notice(
        `The URL names both the entry ${wanted.entry} and the batch ${wanted.batch}; a URL selects one of them. Pick one.`,
      );
      return;
    case "entry": {
      const entry = entries.find(({ id }) => id === wanted.id);
      if (entry === undefined) notice(`No entry ${wanted.id}. Pick one.`);
      else selectEntry(entry, givenOfQuery(query));
      return;
    }
    case "batch": {
      const batch = batches.find(({ id }) => id === wanted.id);
      if (batch === undefined) notice(`No batch ${wanted.id}. Pick one.`);
      else selectBatch(batch);
      return;
    }
  }
};

void start();
