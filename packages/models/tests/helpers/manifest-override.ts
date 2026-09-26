/**
 * 故障注入用の**一時 manifest**（配布形のディレクトリを symlink で借り、`karume.json` だけを
 * 差し替えた取得元）。手筋は `e2e_gemma4_quant_test.ts` の一時 manifest と同じで、元の配布形と
 * 重みは 1 バイトも書き換えない。
 *
 * NOTE: `Deno.symlink` / `Deno.makeTempDir` を使うのはテストだけ（パッケージ本体は Web 標準 API
 * のみ — 横断不変条件）。
 */

import type { DistributionSource } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";

/**
 * `root` の配布形を `manifest` に差し替えた取得元で `body` を回す（一時ディレクトリは抜けたら
 * 消す — symlink を消すだけで、指す先の実体には触れない）。
 */
export const withManifestOverride = async <T>(
  root: URL,
  manifest: unknown,
  body: (source: DistributionSource) => Promise<T>,
): Promise<T> => {
  const temporary = await Deno.makeTempDir({ prefix: "karume-ab-manifest-" });
  try {
    for await (const entry of Deno.readDir(root)) {
      if (entry.name === "karume.json") continue;
      await Deno.symlink(new URL(entry.name, root), `${temporary}/${entry.name}`);
    }
    await Deno.writeTextFile(`${temporary}/karume.json`, JSON.stringify(manifest), {
      createNew: true,
    });
    return await body(denoDirectory(temporary));
  } finally {
    await Deno.remove(temporary, { recursive: true });
  }
};
