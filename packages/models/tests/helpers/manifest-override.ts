/**
 * 故障注入用の**一時 manifest**（配布形のディレクトリを symlink で借り、`karume.json` だけを
 * 差し替えた取得元）。手筋は `e2e_gemma4_quant_test.ts` の一時 manifest と同じで、元の配布形と
 * 重みは 1 バイトも書き換えない。
 *
 * NOTE: `Deno.symlink` / `Deno.makeTempDir` を使うのはテストだけ（パッケージ本体は Web 標準 API
 * のみ — 横断不変条件）。
 */

import { makeTempDir, readDir, remove, symlink, writeTextFile } from "@karume/test-support/fs";
import type { DistributionSource, LocalDirectoryOptions } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";

/**
 * `root` の配布形を `manifest` に差し替えた取得元で `body` を回す（一時ディレクトリは抜けたら
 * 消す — symlink を消すだけで、指す先の実体には触れない）。
 *
 * `options` go to the source as they are — a distribution with cross-repository references needs
 * its `crossRepo` mapping on the overridden source too.
 */
export const withManifestOverride = async <T>(
  root: URL,
  manifest: unknown,
  body: (source: DistributionSource) => Promise<T>,
  options: LocalDirectoryOptions = {},
): Promise<T> => {
  const temporary = await makeTempDir({ prefix: "karume-ab-manifest-" });
  try {
    for await (const entry of readDir(root)) {
      if (entry.name === "karume.json") continue;
      await symlink(new URL(entry.name, root), `${temporary}/${entry.name}`);
    }
    await writeTextFile(`${temporary}/karume.json`, JSON.stringify(manifest), {
      createNew: true,
    });
    return await body(nodeDirectory(temporary, options));
  } finally {
    await remove(temporary, { recursive: true });
  }
};
