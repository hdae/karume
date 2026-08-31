/**
 * `@karume/hub/deno` — Deno のファイルシステムを取得元にするサブパス。
 *
 * MUST: **このサブパスだけが Deno API に依存する**（`Deno.readFile`）。hub 本体（`mod.ts`）の
 * 「ランタイム依存は Web 標準 API のみ」という不変条件への opt-in の carve-out であり、
 * ブラウザ向けのコードから import してはならない — 本体を通る限り `Deno` は 1 度も現れない
 * （ブラウザでは OPFS / IndexedDB / File System Access の picker を
 * {@link ../mod.ts localDirectory} の `DirectoryAdapter` として渡す）。
 *
 * ここに置けるのは「ローカルの実体をどう読むか」だけで、取得元としての意味論（世代・検証・
 * 越境・進捗）は本体の `localDirectory` が持つ。
 */

import {
  type DirectoryAdapter,
  type DistributionSource,
  localDirectory,
  type LocalDirectoryOptions,
} from "./mod.ts";

/**
 * root の下の実体を指す位置。文字列 root は**そのまま継ぐ** — manifest の path は POSIX 相対
 * （`transformer/model.safetensors`）で、`..` は parse 時に弾かれている（`ManifestPathError`）
 * ので、`URL` を通して正規化する必要が無い。逆に `URL` を通すと Windows のドライブ文字を
 * `file:` へ写す綴りが要るため、文字列のままの方が壊れにくい。
 */
const locate = (base: string | URL, path: string): string | URL =>
  typeof base === "string" ? `${base}${path}` : new URL(path, base);

/** 末尾 `/` を保証する（無いと `URL` 解決が兄弟を指し、文字列連結は path がめり込む）。 */
const asDirectory = (root: string | URL): string | URL => {
  const href = typeof root === "string" ? root : root.href;
  const slashed = href.endsWith("/") ? href : `${href}/`;
  return typeof root === "string" ? slashed : new URL(slashed);
};

/**
 * Deno 上のディレクトリを取得元にする。`root` はディレクトリのパス（相対なら cwd 基準）か
 * `file:` URL。
 *
 * ```ts ignore
 * import { loadManifest } from "@karume/hub";
 * import { denoDirectory } from "@karume/hub/deno";
 *
 * const loaded = await loadManifest(denoDirectory("./models/karume-gemma4-e2b"));
 * ```
 *
 * 必要な権限は root 以下への `--allow-read` だけ（network も CacheStorage も通らない）。
 */
export const denoDirectory = (
  root: string | URL,
  options: LocalDirectoryOptions = {},
): DistributionSource => {
  const base = asDirectory(root);
  const adapter: DirectoryAdapter = {
    readFile: async (path, { signal }) => {
      const at = locate(base, path);
      try {
        return await Deno.readFile(at, { ...(signal === undefined ? {} : { signal }) });
      } catch (error) {
        // MUST: 中断はそのまま上げる（共通層が「取り消し」と「取得失敗」を区別できなくなる）。
        if (signal?.aborted === true) throw error;
        // 欠損・権限・実体がディレクトリ、いずれも「読めない」で等価に落とす（fail loudly）。
        // **実体のパス**を名乗るのはここだけの責務 — 共通層が知っているのは manifest 上の
        // 相対 path までで、どのディレクトリの下を探したかはアダプターしか知らない。
        throw new Error(`@karume/hub/deno: ${at} を読めない`, { cause: error });
      }
    },
  };
  return localDirectory(adapter, { label: String(root), ...options });
};
