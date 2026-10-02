/// <reference lib="dom" />
/**
 * GPU lab のページの共通部品（3 タブと `main.ts` が使うヘルパと、タブへ渡す `Lab` の型）。
 *
 * 置くのはヘルパ関数と型だけ — タブの基盤（登録・ライフサイクル）は作らない。タブは
 * `mount(root, …)` の形の関数で、DOM は `root` 配下の `data-ui` 属性で引く（同じ名前の要素が
 * 別のタブにあっても取り違えない）。
 */
import type { GeometryProfile } from "../../../packages/runtime/mod.ts";
import type { GeometryProfileRequested } from "../../anima-residency/record.ts";
import type { ServerConfig } from "../server.ts";

// 型だけの import（bundle で消える — ページに Deno の API は入らない）。
export type { ServerConfig };

export const TIMESTAMP_QUERY = "timestamp-query";

/**
 * GPU 設定の幾何プロファイルの選び方（ADR 0115 追記決定 6 の注入口）。`auto` は注入なし、`default` /
 * `builtin` / `generated` は `acquireGpu({ geometryProfile })` に表を渡す（adapter の vendor / architecture は
 * 照合しない）。`saved-matched` は適用時に取った保存物の文字列（`stored`）を持ち、GPU を取るときに
 * コールバック形で照合して、一致した表だけを注入する（ADR 0117 決定 6 — `injectable-tables.ts` の
 * `resolveSavedProfile`）。`id` は保存物の表の id（読めなければ undefined）。
 */
export type ProfileChoice =
  | { readonly kind: "auto" }
  | { readonly kind: "default"; readonly profile: GeometryProfile }
  | { readonly kind: "builtin"; readonly profile: GeometryProfile }
  | { readonly kind: "generated"; readonly profile: GeometryProfile }
  | { readonly kind: "saved-matched"; readonly stored: string; readonly id?: string };

/** 全タブ共通の GPU 設定（「適用」で確定したもの）。 */
export type GpuSettings = {
  readonly choice: ProfileChoice;
  /** timestamp-query を要求するか（アダプタが列挙しなければ常に false）。 */
  readonly timestamps: boolean;
};

/**
 * 記録に残す「何を頼んだか」の綴り（`auto` / `default` / `builtin:<id>` / `generated:<id>` / `saved:<id>` —
 * tools/anima-residency/record.ts の `GeometryProfileRequested`）。
 */
export const requestedLabel = (choice: ProfileChoice): GeometryProfileRequested => {
  switch (choice.kind) {
    case "auto":
      return "auto";
    case "default":
      return "default";
    case "builtin":
      return `builtin:${choice.profile.id}`;
    case "generated":
      return `generated:${choice.profile.id}`;
    case "saved-matched":
      return `saved:${choice.id ?? "(読めない)"}`;
  }
};

/**
 * `acquireGpu` に表そのものとして渡す表（`auto` は渡さない・`saved-matched` は GPU を取るときにコールバックで
 * 決まるのでここでは渡さない）。
 */
export const injectedProfile = (choice: ProfileChoice): GeometryProfile | undefined =>
  choice.kind === "auto" || choice.kind === "saved-matched" ? undefined : choice.profile;

/** `main.ts` がタブへ渡す共有の面。 */
export type Lab = {
  readonly config: ServerConfig;
  readonly adapterInfo: GPUAdapterInfo;
  /** アダプタが `timestamp-query` を列挙したか。 */
  readonly timestampFeature: boolean;
  /** 適用中の GPU 設定。 */
  readonly settings: () => GpuSettings;
  /**
   * GPU を使う操作の排他（掃引・Anima の操作・GPU 設定の適用を重ねない — 同じ GPU で 2 つ回すと
   * 時間の比べ物にならない）。別の操作が実行中なら投げる。返り値で解く。
   */
  readonly lock: (label: string) => () => void;
};

/** `root` 配下の `[data-ui="name"]`（型が違えば落とす — ページの組み違いを黙って進めない）。 */
export const element = <T extends HTMLElement>(
  root: ParentNode,
  name: string,
  type: new () => T,
): T => {
  const found = root.querySelector(`[data-ui="${name}"]`);
  if (!(found instanceof type)) throw Error(`Missing page element [data-ui="${name}"]`);
  return found;
};

export const setStatus = (target: HTMLElement, text: string): void => {
  target.textContent = text;
};

export const describeError = (error: unknown): { name: string; message: string } => {
  if (error instanceof AggregateError) {
    return {
      name: error.name,
      message: `${error.message} [${
        error.errors.map((cause) => {
          const inner = describeError(cause);
          return `${inner.name}: ${inner.message}`;
        }).join(" | ")
      }]`,
    };
  }
  return error instanceof Error
    ? { name: error.name, message: error.message }
    : { name: "Error", message: String(error) };
};

/** 状態行に出す 1 行（`Error` でない値はそのまま文字列にする）。 */
export const errorText = (error: unknown): string => {
  if (!(error instanceof Error)) return String(error);
  const { name, message } = describeError(error);
  return `${name}: ${message}`;
};

/** adapter の 1 行表示（`GPUAdapterInfo` と掃引の記録の `adapter` のどちらも渡せる）。 */
export const adapterSummary = (
  info: Pick<GPUAdapterInfo, "vendor" | "architecture" | "device" | "description">,
): string =>
  [info.vendor, info.architecture, info.device, info.description].filter((v) => v !== "").join(
    " / ",
  );

/** checkout の版の短い表示（`abcdef12 (dirty)`）。 */
export const checkoutLabel = (config: ServerConfig): string =>
  `${config.revision.slice(0, 8)}${config.dirty ? " (dirty)" : ""}`;

export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** 文字列をファイルとして落とす（ダウンロードが塞がれた置き場ではコピーの口を使う）。 */
export const downloadText = (text: string, fileName: string, type: string): void => {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

/**
 * textarea の中身をクリップボードへ。clipboard API が拒まれたら textarea を選択状態にして
 * false を返す（呼び手が「手でコピー」の案内を出す）。
 */
export const copyFromTextarea = async (textarea: HTMLTextAreaElement): Promise<boolean> => {
  try {
    await navigator.clipboard.writeText(textarea.value);
    return true;
  } catch {
    textarea.focus();
    textarea.select();
    return false;
  }
};
