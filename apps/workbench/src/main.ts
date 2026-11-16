/**
 * The shell: the list of entries, the selected entry's panel (`shell/entry-panel.ts`) and the first
 * selection from the URL. The selected entry, its parameters and the declared backend live in the
 * URL, so a link reproduces a run setup; nothing starts without a click on Run. One run at a time
 * (`shell/run-lock.ts`).
 *
 * SPEC: docs/spec/workbench.md#the-shell
 */

import "./style.css";

import { registeredEntries } from "./entries.ts";
import type { Entry } from "./entry.ts";
import { errorText } from "./record.ts";
import { element } from "./shell/dom.ts";
import { entryPanel } from "./shell/entry-panel.ts";
import { createRunLock } from "./shell/run-lock.ts";
import { BACKEND_KEY, ENTRY_KEY, givenOfQuery } from "./shell/url.ts";

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

const start = (): void => {
  const root = document.getElementById("app");
  if (root === null) throw new Error("index.html has no #app");
  const entries = registeredEntries();
  const query = new URLSearchParams(location.search);
  const backend = element("input", {
    type: "text",
    value: query.get(BACKEND_KEY) ?? "",
    placeholder: "d3d12, metal, vulkan…",
  });
  const listButtons = new Map<Entry, HTMLButtonElement>(
    entries.map((entry): [Entry, HTMLButtonElement] => [
      entry,
      element("button", { type: "button" }, [
        element("strong", { textContent: entry.title }),
        element("span", { textContent: entry.description }),
      ]),
    ]),
  );
  const lock = createRunLock([...listButtons.values()]);
  const nav = element("nav", {}, [
    element(
      "ul",
      { className: "entries" },
      [...listButtons.values()].map((button) => element("li", {}, [button])),
    ),
  ]);
  const layout = element("div", { className: "layout" });
  // Asked on the first selection, once for the page.
  let gpuProblem: Promise<string | undefined> | undefined;
  let selection: AbortController | undefined;

  const select = (entry: Entry, given: Readonly<Record<string, string>>): void => {
    for (const [listed, button] of listButtons) {
      button.setAttribute("aria-current", String(listed === entry));
    }
    selection?.abort();
    selection = new AbortController();
    gpuProblem ??= probeGpu();
    layout.replaceChildren(
      nav,
      entryPanel({ entry, given, lock, backend, gpuProblem, selection: selection.signal }),
    );
  };
  for (const [entry, button] of listButtons) {
    button.addEventListener("click", () => select(entry, {}));
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

  const wanted = query.get(ENTRY_KEY);
  const selected = entries.find(({ id }) => id === wanted);
  if (selected === undefined) {
    layout.replaceChildren(
      nav,
      element("section", {
        className: "panel",
        textContent: wanted === null ? "Pick an entry." : `No entry ${wanted}. Pick one.`,
      }),
    );
    return;
  }
  select(selected, givenOfQuery(query));
};

start();
