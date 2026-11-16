/**
 * The shell: the list of entries, the selected entry's parameter form, Run / Cancel, the status
 * line, and after a run the verdict, the summary, where the record went and the record itself. The
 * selected entry, its parameters and the declared backend live in the URL, so a link reproduces a
 * run setup; nothing starts without a click on Run. One run at a time.
 */

import "./style.css";

import { registeredEntries } from "./entries.ts";
import { type Entry, parameterText, parseParameters, type ParameterValues } from "./entry.ts";
import { readEnvironment } from "./environment.ts";
import { executeEntry } from "./execute.ts";
import { errorText, recordText, type WorkbenchRecord } from "./record.ts";
import { currentCheckout } from "./shell/checkout.ts";
import { element } from "./shell/dom.ts";
import { browserDistributions } from "./shell/distributions.ts";
import { parameterForm } from "./shell/form.ts";
import { downloadRecord, type SaveResult, saveRecord } from "./shell/save.ts";

/** Query keys that are not parameters. */
const RESERVED_KEYS = ["entry", "backend"];

/** The record text shown in the page; the download and the copy are always whole. */
const SHOWN_CHARACTERS = 200_000;

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

const writeUrl = (entry: Entry, values: ParameterValues, backend: string): void => {
  const query = new URLSearchParams({ entry: entry.id });
  for (const parameter of entry.parameters) {
    query.set(parameter.name, parameterText(values[parameter.name]));
  }
  if (backend !== "") query.set("backend", backend);
  history.replaceState(null, "", `?${query.toString()}`);
};

const resultView = (record: WorkbenchRecord, text: string, saved: SaveResult): HTMLElement => {
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

const start = (): void => {
  const root = document.getElementById("app");
  if (root === null) throw new Error("index.html has no #app");
  const entries = registeredEntries();
  const query = new URLSearchParams(location.search);
  const backend = element("input", {
    type: "text",
    value: query.get("backend") ?? "",
    placeholder: "d3d12, metal, vulkan…",
  });
  const panel = element("section", { className: "panel" });
  const listButtons = new Map<Entry, HTMLButtonElement>();
  let running = false;
  let gpuProblem: Promise<string | undefined> | undefined;

  const select = (entry: Entry, given: Readonly<Record<string, string>>): void => {
    for (const [listed, button] of listButtons) {
      button.setAttribute("aria-current", String(listed === entry));
    }
    const status = element("p", { className: "status" });
    let values: ParameterValues;
    try {
      values = parseParameters(entry.parameters, given);
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
    backend.onchange = sync;
    writeUrl(entry, values, backend.value.trim());

    const run = element("button", { type: "button", textContent: "Run" });
    const cancel = element("button", { type: "button", textContent: "Cancel", disabled: true });
    let controller: AbortController | undefined;
    cancel.addEventListener("click", () => controller?.abort());
    const setRunning = (value: boolean): void => {
      running = value;
      run.disabled = value;
      cancel.disabled = !value;
      for (const button of listButtons.values()) button.disabled = value;
    };
    run.addEventListener("click", () => {
      if (running) return;
      let parameters: ParameterValues;
      try {
        parameters = parseParameters(entry.parameters, form.read());
      } catch (error) {
        status.textContent = errorText(error);
        return;
      }
      controller = new AbortController();
      const { signal } = controller;
      setRunning(true);
      result.replaceChildren();
      status.textContent = "Starting…";
      const declared = backend.value.trim();
      void (async () => {
        try {
          const [environment, checkout] = await Promise.all([
            readEnvironment(navigator.gpu, {
              host: "browser",
              userAgent: navigator.userAgent,
              crossOriginIsolated,
              ...(declared === "" ? {} : { backend: { name: declared, source: "declared" } }),
            }),
            currentCheckout(),
          ]);
          const record = await executeEntry({
            entry,
            parameters,
            gpuOptions: {},
            signal,
            progress: (text) => {
              status.textContent = text;
            },
            openDistribution: browserDistributions(document.baseURI),
            environment,
            checkout,
          });
          status.textContent =
            record.outcome === "error"
              ? `Failed after ${record.seconds} s: ${record.error?.split("\n", 1)[0] ?? ""}`
              : `${record.outcome === "done" ? "Finished" : "Cancelled"} after ${record.seconds} s`;
          const text = recordText(record);
          result.replaceChildren(resultView(record, text, await saveRecord(text)));
        } catch (error) {
          status.textContent = `The run could not start: ${errorText(error)}`;
        } finally {
          setRunning(false);
        }
      })();
    });

    gpuProblem ??= probeGpu();
    void gpuProblem.then((problem) => {
      if (problem === undefined) actions.replaceChildren(run, " ", cancel);
      else actions.textContent = problem;
    });

    panel.replaceChildren(
      element("h2", { textContent: entry.title }),
      element("p", { textContent: entry.description }),
      form.element,
      element("label", { className: "backend" }, ["backend (optional) ", backend]),
      actions,
      status,
      result,
    );
  };

  const list = element(
    "ul",
    { className: "entries" },
    entries.map((entry) => {
      const button = element("button", { type: "button" }, [
        element("strong", { textContent: entry.title }),
        element("span", { textContent: entry.description }),
      ]);
      button.addEventListener("click", () => select(entry, {}));
      listButtons.set(entry, button);
      return element("li", {}, [button]);
    }),
  );

  root.replaceChildren(
    element("header", {}, [
      element("h1", { textContent: "Karume workbench" }),
      element("p", {
        textContent: import.meta.env.DEV
          ? "Records are saved under outputs/workbench/ by the dev server."
          : "A production build saves no records: download them.",
      }),
    ]),
    element("div", { className: "layout" }, [element("nav", {}, [list]), panel]),
  );

  const selected = entries.find(({ id }) => id === query.get("entry"));
  if (selected === undefined) {
    panel.textContent =
      query.get("entry") === null ? "Pick an entry." : `No entry ${query.get("entry")}. Pick one.`;
    return;
  }
  select(
    selected,
    Object.fromEntries([...query].filter(([name]) => !RESERVED_KEYS.includes(name))),
  );
};

start();
