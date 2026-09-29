/// <reference lib="dom" />
/**
 * GPU lab のページの入口（`../server.ts` が起動時に bundle する — ADR 0115 追記決定 6 の PoC）。
 *
 * 「掃引（ベンチマーク）→ 幾何プロファイルの生成（最適化）→ 注入して Anima を実行」を 1 ページで回す。
 * ここが持つのはタブの切替・全タブ共通の環境行・**GPU 設定**（幾何プロファイルの注入と timestamp の
 * 要求 — 「適用」で確定）・GPU 操作の排他だけ。各タブは `mount(root, …)` の形のモジュール:
 * `sweep-tab.ts`・`profile-tab.ts`・`anima-tab.ts`。
 *
 * GPU 設定の効き方: Anima のタブの GPU は適用中の設定で取る（注入があれば adapter を見ずにその表を使う）。
 * 掃引のタブは各幾何を明示して測るので注入は効かない（timestamp の要求だけが効く）。適用は Anima の
 * pipeline・ダミー・GPU を畳み、GPU を持っていたなら新しい設定で取り直す。
 */
import type { GeometryProfile } from "../../../packages/runtime/mod.ts";
import {
  DEFAULT_GEOMETRY_PROFILE,
  selectGeometryProfile,
} from "../../../packages/runtime/src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../../../packages/runtime/src/kernels/geometry-profiles/index.ts";
import { mountAnimaTab } from "./anima-tab.ts";
import {
  adapterSummary,
  checkoutLabel,
  element,
  errorText,
  type GpuSettings,
  type Lab,
  type ProfileChoice,
  requestedLabel,
  type ServerConfig,
  setStatus,
  TIMESTAMP_QUERY,
} from "./common.ts";
import { mountProfileTab, type ProfileTab } from "./profile-tab.ts";
import { mountSweepTab } from "./sweep-tab.ts";

const TABS = ["sweep", "profile", "anima"] as const;
type Tab = typeof TABS[number];

const isTab = (value: string): value is Tab => (TABS as readonly string[]).includes(value);

/** select の値（`builtin:` を前置するのは、埋め込みの id が `auto` などの綴りと重ならないように）。 */
const BUILTIN_PREFIX = "builtin:";
const GENERATED = "generated";

const header = element(document, "header", HTMLElement);
const ui = {
  profile: element(header, "profile", HTMLSelectElement),
  timestamps: element(header, "timestamps", HTMLInputElement),
  apply: element(header, "apply", HTMLButtonElement),
  pending: element(header, "pending", HTMLElement),
  gpuStatus: element(header, "gpu-status", HTMLElement),
  environment: element(header, "environment", HTMLElement),
};

const tabRoot = (tab: Tab): HTMLElement => {
  const found = document.querySelector(`[data-tab="${tab}"]`);
  if (!(found instanceof HTMLElement)) throw Error(`Missing tab panel [data-tab="${tab}"]`);
  return found;
};

const tabButtons = (): HTMLButtonElement[] => [
  ...header.querySelectorAll<HTMLButtonElement>("button[data-tab-button]"),
];

const gpuStatus = (text: string): void => setStatus(ui.gpuStatus, text);

const initialize = async (): Promise<void> => {
  // 相対 path にするのは、同じページと bundle を静的な置き場にそのまま載せるため。
  const configResponse = await fetch("config.json");
  if (!configResponse.ok) throw Error(`config.json HTTP ${configResponse.status}`);
  const config: ServerConfig = await configResponse.json();
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) {
    throw Error("ハードウェア WebGPU を使える Chrome が必要です");
  }
  const adapterInfo = adapter.info;
  const timestampFeature = adapter.features.has(TIMESTAMP_QUERY);

  const state: {
    /** 適用中の GPU 設定。 */
    settings: GpuSettings;
    /** プロファイルのタブが直近に作った表（作り直しを始めたら消す）。 */
    generated?: GeometryProfile;
    /** 実行中の GPU 操作（{@link Lab.lock}）。 */
    activity?: string;
    tab: Tab;
  } = { settings: { choice: { kind: "auto" }, timestamps: timestampFeature }, tab: "sweep" };

  /**
   * 「生成した表」の選択肢が指す表: 直近に作った表、無ければ適用中の生成した表（使用中なので、作り直しが
   * 失敗しても選択肢から消さない）。
   */
  const generatedOption = (): GeometryProfile | undefined => {
    const applied = state.settings.choice;
    return state.generated ?? (applied.kind === "generated" ? applied.profile : undefined);
  };

  /** 選び方 → select の値。 */
  const selectValue = (choice: ProfileChoice): string =>
    choice.kind === "builtin"
      ? `${BUILTIN_PREFIX}${choice.profile.id}`
      : choice.kind === "generated"
      ? GENERATED
      : choice.kind;

  /** select の値 → 選び方。 */
  const readChoice = (): ProfileChoice => {
    const value = ui.profile.value;
    if (value === "auto") return { kind: "auto" };
    if (value === "default") return { kind: "default", profile: DEFAULT_GEOMETRY_PROFILE };
    if (value === GENERATED) {
      const profile = generatedOption();
      if (profile === undefined) throw Error("生成した表がまだ無い");
      return { kind: "generated", profile };
    }
    const id = value.slice(BUILTIN_PREFIX.length);
    const profile = BUILTIN_GEOMETRY_PROFILES.find((entry) => entry.id === id);
    if (!value.startsWith(BUILTIN_PREFIX) || profile === undefined) {
      throw Error(`幾何プロファイルの選択 ${value} を知らない`);
    }
    return { kind: "builtin", profile };
  };

  /** 選択が適用中と違うか（生成した表は作り直すと中身が変わるので、同じ id でも表そのもので比べる）。 */
  const pending = (): boolean => {
    const applied = state.settings;
    if (ui.timestamps.checked !== applied.timestamps) return true;
    const choice = readChoice();
    return choice.kind !== applied.choice.kind ||
      (choice.kind !== "auto" && applied.choice.kind !== "auto" &&
        choice.profile !== applied.choice.profile);
  };

  const renderEnvironment = (): void => {
    const { choice, timestamps } = state.settings;
    // adapterInfo は description ごと渡す（description で照合する表の選択は runtime と同じ）
    const profile = choice.kind === "auto"
      ? `自動 → ${selectGeometryProfile(adapterInfo).id}`
      : `${requestedLabel(choice)}（注入）`;
    ui.environment.textContent = `${adapterSummary(adapterInfo)} · GPU 時間 ${
      timestamps ? "採る" : "採らない"
    }${timestampFeature ? "" : `（${TIMESTAMP_QUERY} 無し）`} · 幾何プロファイル ${profile}${
      state.tab === "sweep" ? "（掃引は明示幾何なので掃引の結果には効かない）" : ""
    } · ${checkoutLabel(config)}`;
    ui.pending.textContent = pending() ? "未適用の変更があります" : "";
  };

  const lock = (label: string): () => void => {
    if (state.activity !== undefined) throw Error(`${state.activity}の実行中 — 終わってから`);
    state.activity = label;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      state.activity = undefined;
    };
  };

  const lab: Lab = {
    config,
    adapterInfo,
    timestampFeature,
    settings: () => state.settings,
    lock,
  };

  // 掃引が終わったらプロファイルのタブの「直近の結果」を今に合わせる（タブを開いたままでも）
  const sweep = mountSweepTab(tabRoot("sweep"), lab, () => profile.refresh());
  const anima = mountAnimaTab(tabRoot("anima"), lab);

  /**
   * 「生成した表」の選択肢を今の状態に合わせる: 指す表が無ければ消し（選んでいたなら選択を適用中の値へ
   * 戻す）、適用中の表なら文言に「（適用中）」を付ける。
   */
  const renderGeneratedOption = (): void => {
    const profile = generatedOption();
    let option = ui.profile.querySelector<HTMLOptionElement>(`option[value="${GENERATED}"]`);
    if (profile === undefined) {
      if (option === null) return;
      const selected = ui.profile.value === GENERATED;
      option.remove();
      if (selected) ui.profile.value = selectValue(state.settings.choice);
      return;
    }
    if (option === null) {
      option = document.createElement("option");
      option.value = GENERATED;
      ui.profile.append(option);
    }
    const applied = state.settings.choice;
    option.textContent = `生成した表: ${profile.id}（注入）${
      applied.kind === "generated" && applied.profile === profile ? "（適用中）" : ""
    }`;
  };

  /**
   * `select` を渡すと、排他を取った後に GPU 設定の選択をその値にしてから適用し、適用に失敗したら選択を
   * 適用中の値へ戻す（排他が取れなければ選択に触らない — 利用者の未適用の選択を上書きしない）。
   * 失敗は GPU 設定の状態行に出してから投げ直す（プロファイルのタブから呼ばれたときはそちらにも出る）。
   */
  const apply = async (select?: string): Promise<void> => {
    let release: (() => void) | undefined;
    let selected = false;
    try {
      release = lock("GPU 設定の適用");
      if (select !== undefined) {
        ui.profile.value = select;
        selected = true;
      }
      const next: GpuSettings = {
        choice: readChoice(),
        timestamps: timestampFeature && ui.timestamps.checked,
      };
      const previous = state.settings;
      gpuStatus("適用中 …");
      const held = await anima.reset();
      state.settings = next;
      renderGeneratedOption();
      renderEnvironment();
      if (held) {
        try {
          await anima.acquire();
        } catch (error) {
          // 取れなかった設定を適用中として残すと、以後の Anima の操作が同じ理由で落ち続ける
          state.settings = previous;
          renderGeneratedOption();
          renderEnvironment();
          throw error;
        }
      }
      gpuStatus(
        `適用しました（幾何プロファイル ${requestedLabel(next.choice)} · GPU 時間 ${
          next.timestamps ? "採る" : "採らない"
        }${held ? " · Anima の GPU を取り直しました" : ""}）`,
      );
    } catch (error) {
      if (selected) {
        ui.profile.value = selectValue(state.settings.choice);
        renderEnvironment();
      }
      gpuStatus(errorText(error));
      throw error;
    } finally {
      release?.();
    }
  };

  const offerGenerated = (profile: GeometryProfile): void => {
    state.generated = profile;
    renderGeneratedOption();
    renderEnvironment();
  };

  const withdrawGenerated = (): void => {
    state.generated = undefined;
    renderGeneratedOption();
    renderEnvironment();
  };

  const profile: ProfileTab = mountProfileTab(tabRoot("profile"), {
    adapterInfo,
    latestSweep: sweep.latestReport,
    offerGenerated,
    withdrawGenerated,
    applyGenerated: () => apply(GENERATED),
  });

  const showTab = (tab: Tab): void => {
    state.tab = tab;
    for (const name of TABS) tabRoot(name).hidden = name !== tab;
    for (const button of tabButtons()) {
      button.setAttribute("aria-selected", String(button.dataset.tabButton === tab));
    }
    if (tab === "profile") profile.refresh();
    renderEnvironment();
    history.replaceState(null, "", `#${tab}`);
  };

  const fillProfileOptions = (): void => {
    const option = (value: string, text: string): HTMLOptionElement => {
      const created = document.createElement("option");
      created.value = value;
      created.textContent = text;
      return created;
    };
    ui.profile.replaceChildren(
      option("auto", `自動 — adapter で選ぶ（→ ${selectGeometryProfile(adapterInfo).id}）`),
      option("default", "default（既定の表を注入）"),
      // 埋め込みの全表（match を省いた注入専用の表も — 自動では選ばれないので、ここが使う入口）
      ...BUILTIN_GEOMETRY_PROFILES.map(({ id, match }) =>
        option(
          `${BUILTIN_PREFIX}${id}`,
          match === undefined
            ? `${id}（注入専用）`
            : match.description !== undefined
            ? `${id}（注入 — ${match.description} 用）`
            : `${id}（注入 — ${
              [match.vendor, match.architecture].filter((v) => v !== undefined).join(" / ")
            } 用）`,
        )
      ),
    );
    ui.profile.value = "auto";
  };

  fillProfileOptions();
  ui.timestamps.checked = timestampFeature;
  ui.timestamps.disabled = !timestampFeature;
  gpuStatus(
    timestampFeature
      ? "準備完了。GPU 設定は「適用」で確定します（今は自動・GPU 時間を採る）。"
      : `このアダプタは ${TIMESTAMP_QUERY} を持たないので、掃引は壁時計で測り、Anima の GPU 時間は採れません。`,
  );
  ui.profile.addEventListener("change", renderEnvironment);
  ui.timestamps.addEventListener("change", renderEnvironment);
  // 失敗は apply が GPU 設定の状態行に出している（ここで重ねて出さない）
  ui.apply.addEventListener("click", () => void apply().catch(() => undefined));
  for (const button of tabButtons()) {
    button.addEventListener("click", () => {
      const tab = button.dataset.tabButton ?? "";
      if (isTab(tab)) showTab(tab);
    });
  }
  const initial = location.hash.slice(1);
  showTab(isTab(initial) ? initial : "sweep");
};

initialize().catch((error: unknown) => gpuStatus(errorText(error)));
