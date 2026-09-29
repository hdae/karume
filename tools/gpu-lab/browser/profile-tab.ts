/// <reference lib="dom" />
/**
 * プロファイルのタブ — 掃引の記録から幾何プロファイル（ADR 0115 の静的な表）を作り、GPU 設定へ
 * 渡して注入できるようにする。
 *
 * 材料は掃引タブの直近の結果（メモリ上の記録）と、読み込んだ / 貼り付けた掃引の JSON（複数可）。規則は
 * CLI（`tools/geometry-sweep/main.ts profile`）と同じ純関数（`tools/geometry-sweep/derive.ts`）で、
 * 入力の門（GPU の timestamp で測った掃引だけ・adapter を混ぜない・既定の行が今の runtime の既定）も
 * 同じ — 拒まれたらその理由を状態行へそのまま出す（fail loudly）。
 *
 * 出すもの: 欄ごとの採否と退けた理由の表・TS の生成物（整形前 — 整形は CLI の `deno fmt`）・アプリ用の
 * TS（`@karume/runtime` の型で書いた定数 — アプリが `acquireGpu({ geometryProfile })` へ渡す）・注入に
 * 使う JSON（`acquireGpu({ geometryProfile })` の値）・リポへ登録するコマンド（CLI の `profile` —
 * 生成物を整形して書き、`geometry-profiles/index.ts` へ足す行を案内する）。
 */
import type { GeometryProfile } from "../../../packages/runtime/mod.ts";
import {
  buildGeometryProfile,
  DEFAULT_MIN_SPEEDUP,
  deriveProfile,
  formatRatio,
  type GeneratedProfile,
  parseSweepReport,
  PROFILE_ID,
  profileJson,
  type ProfileSpec,
  regenerateCommand,
  renderAppProfileSource,
  renderProfileSource,
  type SlotVerdict,
  type SweepSource,
} from "../../geometry-sweep/derive.ts";
import type { Report } from "../../geometry-sweep/report.ts";
import {
  adapterSummary,
  copyFromTextarea,
  downloadText,
  element,
  errorText,
  setStatus,
  sha256Hex,
} from "./common.ts";
import { sweepFileName, sweepJson } from "./sweep-tab.ts";

/** 生成物の置き場（CLI の `--out` の既定の親 — ファイル名は `<id>.ts`）。 */
const PROFILE_DIRECTORY = "packages/runtime/src/kernels/geometry-profiles";

/**
 * 記録の path の既定の親（README が勧めるブラウザの記録の置き場）。ページはファイルの本当の置き場を
 * 知らないので、生成物の `provenance` と登録のコマンドはこの下に置いた前提の path を書く。
 */
const SWEEP_DIRECTORY = "outputs/bench-browser";

const ENCODER = new TextEncoder();
// CLI（`profile.ts`）と同じ既定の decoder（BOM を落とす）— 同じファイルから同じ本文と sha256 を得る
const DECODER = new TextDecoder();

export type ProfileTabDeps = {
  readonly adapterInfo: GPUAdapterInfo;
  /** 掃引タブの直近の結果。 */
  readonly latestSweep: () => Report | undefined;
  /** 生成した表を GPU 設定の選択肢に出す。 */
  readonly offerGenerated: (profile: GeometryProfile) => void;
  /** 生成をやり直す前に、GPU 設定の選択肢から直前の表を下げる（適用中なら残す）。 */
  readonly withdrawGenerated: () => void;
  /** 生成した表を GPU 設定で選んで適用する。 */
  readonly applyGenerated: () => Promise<void>;
};

export type ProfileTab = {
  /** 掃引タブの直近の結果の表示を今に合わせる（タブを開いたときに呼ぶ）。 */
  readonly refresh: () => void;
};

type LoadedSource = { readonly key: number; readonly source: SweepSource };

type Generated = {
  readonly spec: ProfileSpec;
  readonly profile: GeneratedProfile;
  readonly outputs: {
    readonly ts: string;
    readonly app: string;
    readonly json: string;
    readonly command: string;
  };
};

type OutputKind = keyof Generated["outputs"];

const isOutputKind = (value: string): value is OutputKind =>
  value === "ts" || value === "app" || value === "json" || value === "command";

/** adapter から id の既定を作る（kebab-case・英小文字始まり — 生成器の id の規則）。 */
const autoId = (adapter: { readonly vendor: string; readonly architecture: string }): string => {
  const slug = [adapter.vendor, adapter.architecture].filter((part) => part !== "").join("-")
    .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return /^[a-z]/.test(slug) ? slug : `gpu-${slug}`;
};

export const mountProfileTab = (root: HTMLElement, deps: ProfileTabDeps): ProfileTab => {
  const ui = {
    useLatest: element(root, "use-latest", HTMLInputElement),
    latestLabel: element(root, "latest-label", HTMLElement),
    files: element(root, "files", HTMLInputElement),
    paste: element(root, "paste", HTMLTextAreaElement),
    addPaste: element(root, "add-paste", HTMLButtonElement),
    sources: element(root, "sources", HTMLTableSectionElement),
    id: element(root, "id", HTMLInputElement),
    vendor: element(root, "vendor", HTMLInputElement),
    architecture: element(root, "architecture", HTMLInputElement),
    vendorOnly: element(root, "vendor-only", HTMLInputElement),
    minSpeedup: element(root, "min-speedup", HTMLElement),
    generate: element(root, "generate", HTMLButtonElement),
    apply: element(root, "apply", HTMLButtonElement),
    status: element(root, "status", HTMLElement),
    verdicts: element(root, "verdicts", HTMLTableSectionElement),
    outputKind: element(root, "output-kind", HTMLSelectElement),
    output: element(root, "output", HTMLTextAreaElement),
    copyOutput: element(root, "copy-output", HTMLButtonElement),
    saveOutput: element(root, "save-output", HTMLButtonElement),
  };

  const state: {
    loaded: LoadedSource[];
    nextKey: number;
    generated?: Generated;
    /** 直近の結果の表示に使った記録（新しい掃引が来たら既定でチェックを入れ直す）。 */
    shownLatest?: Report;
  } = { loaded: [], nextKey: 1 };

  const status = (text: string): void => setStatus(ui.status, text);

  /**
   * バイト列を掃引の記録として読む（門で拒まれたら投げる — 理由は derive.ts の文言のまま）。sha256 は
   * 渡したバイト列そのもので取る（ファイルは読んだ bytes — CLI と同じ値）。
   */
  const sourceFromBytes = async (
    bytes: Uint8Array<ArrayBuffer>,
    name: string,
  ): Promise<SweepSource> => {
    const path = `${SWEEP_DIRECTORY}/${name}`;
    let parsed: unknown;
    try {
      parsed = JSON.parse(DECODER.decode(bytes));
    } catch (cause) {
      throw Error(`${path}: JSON として読めない（${errorText(cause)}）`);
    }
    return parseSweepReport(parsed, { path, sha256: await sha256Hex(bytes) });
  };

  /** 生成に使う記録の並び（直近の結果 → 読み込んだ順 — この順が生成物の provenance の順）。 */
  const collectSources = async (): Promise<SweepSource[]> => {
    const sources: SweepSource[] = [];
    if (ui.useLatest.checked) {
      const report = deps.latestSweep();
      if (report === undefined) {
        throw Error("掃引タブにまだ結果が無い（チェックを外すか、掃引を回す）");
      }
      sources.push(await sourceFromBytes(ENCODER.encode(sweepJson(report)), sweepFileName(report)));
    }
    sources.push(...state.loaded.map((entry) => entry.source));
    if (sources.length === 0) {
      throw Error("掃引の記録が無い（掃引タブの直近の結果を使うか、JSON を読み込む / 貼り付ける）");
    }
    return sources;
  };

  /** 表の相手の adapter（最初の記録・無ければこのページの adapter）から id / vendor / architecture の既定。 */
  const autoAdapter = (): { readonly vendor: string; readonly architecture: string } =>
    (ui.useLatest.checked ? deps.latestSweep()?.adapter : undefined) ??
      state.loaded[0]?.source.adapter ?? deps.adapterInfo;

  const renderPlaceholders = (): void => {
    const adapter = autoAdapter();
    ui.id.placeholder = `${autoId(adapter)}（自動）`;
    ui.vendor.placeholder = `${adapter.vendor}（自動）`;
    ui.architecture.placeholder = adapter.architecture === ""
      ? "（自動 — 空なので vendor だけで当てる）"
      : `${adapter.architecture}（自動）`;
  };

  const readSpec = (sources: readonly SweepSource[]): ProfileSpec => {
    const adapter = sources[0].adapter;
    const id = ui.id.value.trim() || autoId(adapter);
    if (!PROFILE_ID.test(id)) throw Error(`id は英小文字始まりの kebab-case（${id}）`);
    const vendor = ui.vendor.value.trim() || adapter.vendor;
    if (vendor === "") throw Error("vendor が空（入力するか、adapter の vendor を持つ記録を使う）");
    const typed = ui.architecture.value.trim() || adapter.architecture;
    const architecture = ui.vendorOnly.checked || typed === "" ? undefined : typed;
    return {
      from: sources.map((source) => source.path),
      id,
      vendor,
      ...(architecture === undefined ? {} : { architecture }),
      out: `${PROFILE_DIRECTORY}/${id}.ts`,
      minSpeedup: DEFAULT_MIN_SPEEDUP,
    };
  };

  const cell = (text: string, className?: string): HTMLTableCellElement => {
    const td = document.createElement("td");
    td.textContent = text;
    if (className !== undefined) td.className = className;
    return td;
  };

  const renderSources = (): void => {
    ui.sources.replaceChildren(
      ...state.loaded.map(({ key, source }) => {
        const tr = document.createElement("tr");
        const remove = document.createElement("button");
        remove.textContent = "外す";
        remove.addEventListener("click", () => {
          state.loaded = state.loaded.filter((entry) => entry.key !== key);
          renderSources();
          renderPlaceholders();
        });
        const action = document.createElement("td");
        action.append(remove);
        const { vendor, architecture, description } = source.adapter;
        tr.append(
          cell(source.path, "wrap"),
          cell([vendor, architecture, description].filter((part) => part !== "").join(" / ")),
          cell(source.date),
          cell(String(source.rows.length), "num"),
          cell(source.sha256.slice(0, 12)),
          action,
        );
        return tr;
      }),
    );
  };

  const addSource = async (bytes: Uint8Array<ArrayBuffer>, name: string): Promise<void> => {
    const source = await sourceFromBytes(bytes, name);
    state.loaded.push({ key: state.nextKey++, source });
    renderSources();
    renderPlaceholders();
    status(`${source.path} を足しました（${source.rows.length} 行）。`);
  };

  const verdictRow = (verdict: SlotVerdict): HTMLTableRowElement => {
    const tr = document.createElement("tr");
    const { outcome } = verdict;
    const rejected = document.createElement("td");
    rejected.className = "wrap";
    if (verdict.rejected.length === 0) rejected.textContent = "—";
    else {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `${verdict.rejected.length} 本`;
      const list = document.createElement("ul");
      for (const { name, reason } of verdict.rejected) {
        const item = document.createElement("li");
        item.textContent = `${name}: ${reason}`;
        list.append(item);
      }
      details.append(summary, list);
      rejected.append(details);
    }
    tr.append(
      cell(verdict.slot),
      cell(verdict.scope, "wrap"),
      cell(String(verdict.cases.length), "num"),
      outcome.kind === "adopted"
        ? cell(`採用 ${outcome.name}`, "good")
        : cell(`既定 ${outcome.name} のまま（${outcome.reason}）`, "wrap"),
      cell(
        outcome.kind === "adopted"
          ? `${formatRatio(outcome.geomean)}（${formatRatio(outcome.min)}〜${
            formatRatio(outcome.max)
          }）`
          : "—",
        "num",
      ),
      rejected,
    );
    return tr;
  };

  const renderOutput = (): void => {
    const kind = ui.outputKind.value;
    if (!isOutputKind(kind)) throw Error(`出力の種類 ${kind} を知らない`);
    const generated = state.generated;
    ui.output.value = generated === undefined ? "" : generated.outputs[kind];
    ui.copyOutput.disabled = generated === undefined;
    ui.saveOutput.disabled = generated === undefined;
    ui.apply.disabled = generated === undefined;
  };

  /** 登録の手順（記録の置き場 → CLI の実行）。コマンドは CLI の再生成コマンドと同じ綴り。 */
  const registrationText = (spec: ProfileSpec): string =>
    [
      "# 1. 掃引の記録を次の path に置く（掃引タブの「JSON を保存」・読み込んだファイル）:",
      ...spec.from.map((path) => `#    ${path}`),
      "# 2. リポ直下で実行する（deno fmt で整形した生成物を書き、index.ts へ足す行を案内する）:",
      ...regenerateCommand(spec),
      "",
    ].join("\n");

  const generate = async (): Promise<void> => {
    state.generated = undefined;
    // 失敗したときに前の表を「生成した表」として残さない（残すとヘッダとタブで食い違う）
    deps.withdrawGenerated();
    ui.verdicts.replaceChildren();
    renderOutput();
    const sources = await collectSources();
    const spec = readSpec(sources);
    const verdicts = deriveProfile(sources, spec);
    const profile = buildGeometryProfile(spec, sources, verdicts);
    state.generated = {
      spec,
      profile,
      outputs: {
        ts: renderProfileSource(spec, sources, verdicts),
        app: renderAppProfileSource(profile),
        json: profileJson(profile),
        command: registrationText(spec),
      },
    };
    ui.verdicts.replaceChildren(...verdicts.map(verdictRow));
    renderOutput();
    deps.offerGenerated(profile);
    const adopted = verdicts.filter((verdict) => verdict.outcome.kind === "adopted").length;
    status(
      `表 ${spec.id} を作りました（採用 ${adopted} 欄 · 既定のまま ${
        verdicts.length - adopted
      } 欄 · 記録 ${sources.length} 本）。「この表を適用」で GPU 設定に注入します。`,
    );
  };

  const saveOutput = (): void => {
    const generated = state.generated;
    const kind = ui.outputKind.value;
    if (generated === undefined || !isOutputKind(kind)) return;
    const { id } = generated.spec;
    const [name, type] = kind === "ts"
      ? [`${id}.ts`, "text/plain"]
      : kind === "app"
      ? [`geometry-profile-${id}.ts`, "text/plain"]
      : kind === "json"
      ? [`geometry-profile-${id}.json`, "application/json"]
      : [`geometry-profile-${id}.sh`, "text/plain"];
    downloadText(generated.outputs[kind], name, type);
  };

  const guarded = (action: () => Promise<void>) => async (): Promise<void> => {
    try {
      await action();
    } catch (error) {
      status(errorText(error));
    }
  };

  const refresh = (): void => {
    const report = deps.latestSweep();
    ui.useLatest.disabled = report === undefined;
    if (report !== state.shownLatest) ui.useLatest.checked = report !== undefined;
    state.shownLatest = report;
    ui.latestLabel.textContent = report === undefined
      ? "（まだ無い — 掃引タブで回すとここに出ます）"
      : `${sweepFileName(report)}（${report.rows.length} 行 · ${
        adapterSummary(report.adapter)
      } · 単位 ${report.gpuTiming.unit}${
        report.gpuTiming.quantized ? " · 100 µs 量子化の疑い" : ""
      }）`;
    renderPlaceholders();
  };

  ui.minSpeedup.textContent = formatRatio(DEFAULT_MIN_SPEEDUP);
  ui.files.addEventListener(
    "change",
    guarded(async () => {
      const files = [...(ui.files.files ?? [])];
      ui.files.value = "";
      for (const file of files) {
        await addSource(new Uint8Array(await file.arrayBuffer()), file.name);
      }
    }),
  );
  ui.addPaste.addEventListener(
    "click",
    guarded(async () => {
      const text = ui.paste.value;
      if (text.trim() === "") throw Error("貼り付け欄が空");
      let date = "";
      try {
        const parsed: unknown = JSON.parse(text);
        if (typeof parsed === "object" && parsed !== null && "date" in parsed) {
          date = String(parsed.date);
        }
      } catch {
        // 読めない JSON は addSource が理由つきで落とす
      }
      // 貼り付けにはファイル名が無い — ページの「JSON を保存」と同じ名前を path の既定にする
      await addSource(
        ENCODER.encode(text),
        `geometry-sweep-browser-${date.replaceAll(":", "-") || "pasted"}.json`,
      );
      ui.paste.value = "";
    }),
  );
  ui.useLatest.addEventListener("change", renderPlaceholders);
  ui.generate.addEventListener("click", guarded(generate));
  ui.apply.addEventListener("click", guarded(deps.applyGenerated));
  ui.outputKind.addEventListener("change", renderOutput);
  ui.copyOutput.addEventListener(
    "click",
    guarded(async () => {
      status(
        await copyFromTextarea(ui.output)
          ? "クリップボードへコピーしました。"
          : "クリップボードが使えないので、選択した内容を手でコピーしてください。",
      );
    }),
  );
  ui.saveOutput.addEventListener("click", saveOutput);
  refresh();
  renderOutput();
  status("材料の掃引を選んで「表を作る」。");
  return { refresh };
};
