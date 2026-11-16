/**
 * Runs one workbench entry under Node + Dawn, as the browser shell does, and writes its record
 * where the dev server would:
 *
 *     flock /tmp/karume-gpu.lock node apps/workbench/run.ts kernel-check [--name=value …]
 *     node apps/workbench/run.ts --list
 *
 * The entry modules are the browser's own. Their imports use Vite features (`import.meta.glob` in
 * the registry, `.wgsl?raw` in `@karume/kernel-checks`), so they are loaded through Vite's module
 * runner — the `ssr` environment of a dev server in middleware mode, which runs modules in this
 * process. The command's body (`node-run.ts`) is loaded through the same runner: were it loaded by
 * Node itself, the host side and the entries would each hold their own copy of the shared modules.
 *
 * Exit code: 0 for a run that is done with verdict pass, incomplete or none; 1 for fail, error or
 * cancelled (SIGINT cancels); 2 for a command line that names no entry or a parameter that does not
 * fit.
 */

import { fileURLToPath } from "node:url";
import { createServer, isRunnableDevEnvironment } from "vite-plus";

const WORKBENCH = fileURLToPath(new URL("./", import.meta.url));

const main = async (args: readonly string[]): Promise<number> => {
  // No HTTP server, no HMR socket and no watcher: only the module runner of the `ssr` environment.
  const server = await createServer({
    configFile: false,
    root: WORKBENCH,
    logLevel: "warn",
    appType: "custom",
    server: { middlewareMode: true, hmr: false, ws: false, watch: null },
  });
  try {
    const environment = server.environments.ssr;
    if (!isRunnableDevEnvironment(environment)) {
      throw new Error("Vite's ssr environment cannot run modules in this process");
    }
    const body = await environment.runner.import<typeof import("./node-run.ts")>("/node-run.ts");
    return await body.runCommand(args);
  } finally {
    await server.close();
  }
};

if (import.meta.main) {
  let code: number;
  try {
    code = await main(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(
      `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
    );
    code = 1;
  }
  // Dawn's GPU object keeps the event loop alive while it is referenced, so the process ends here
  // instead of waiting for it (the command's body flushed stdout and stderr).
  process.exit(code);
}
