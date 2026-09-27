/**
 * タイル幾何の掃引（perf-ledger K-70）を Chrome（WebGPU）で回すページのローカルサーバ。
 *
 * 配信の形は `tools/anima-residency/browser/server.ts` と同じ（localhost 限定・COOP / COEP / CORP・
 * 起動時に `deno bundle --platform browser`）。入力は合成データなので、モデルは配らない — 配る
 * のはページ・bundle・`/config.json` の 3 つだけ。
 */
import { fileResponse } from "../../llm-speed/browser/server.ts";
import { readCheckout } from "../../anima-residency/browser/server.ts";

const headers = (): Headers =>
  new Headers({
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
  });

/** `/config.json` の中身（ページが書き出す JSON にそのまま載る）。 */
export type ServerConfig = {
  readonly revision: string;
  readonly dirty: boolean;
  readonly bundleSha256: string;
};

export const createHandler = (
  bundle: Uint8Array<ArrayBuffer>,
  config: ServerConfig,
): (req: Request) => Promise<Response> => {
  const staticRoot = decodeURIComponent(new URL(".", import.meta.url).pathname);
  return async (req) => {
    const url = new URL(req.url);
    if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) {
      return new Response("Localhost only", { status: 403 });
    }
    if (req.method !== "GET" && req.method !== "HEAD") return new Response(null, { status: 405 });
    try {
      const h = headers();
      if (url.pathname === "/config.json") return Response.json(config, { headers: h });
      if (url.pathname === "/runner.js") {
        h.set("Content-Type", "text/javascript");
        return new Response(bundle, { headers: h });
      }
      if (url.pathname === "/") return await fileResponse(req, `${staticRoot}/index.html`);
      return new Response(null, { status: 404, headers: h });
    } catch (error) {
      console.error(error);
      return new Response("Request failed; see server log", { status: 400, headers: headers() });
    }
  };
};

const main = async (): Promise<void> => {
  if (Deno.args.includes("--help")) {
    console.log("deno task bench:geometry-browser [--port 8789]");
    return;
  }
  const args = new Map<string, string>();
  for (let i = 0; i < Deno.args.length; i += 2) {
    const key = Deno.args[i], value = Deno.args[i + 1];
    if (key !== "--port" || value === undefined || value.startsWith("--") || args.has(key)) {
      throw Error(`Invalid option ${key}`);
    }
    args.set(key, value);
  }
  const port = Number(args.get("--port") ?? 8789);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw Error("Invalid port");
  const build = await Deno.makeTempDir({ prefix: "karume-geometry-sweep-" });
  try {
    const output = `${build}/runner.js`;
    const command = new Deno.Command(Deno.execPath(), {
      args: [
        "bundle",
        "--platform",
        "browser",
        "--output",
        output,
        decodeURIComponent(new URL("runner.ts", import.meta.url).pathname),
      ],
      stdout: "inherit",
      stderr: "inherit",
    });
    if (!(await command.output()).success) throw Error("Browser bundle failed: runner.ts");
    const bundle = await Deno.readFile(output);
    const config: ServerConfig = {
      ...await readCheckout(),
      bundleSha256: Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", bundle)),
        (v) => v.toString(16).padStart(2, "0"),
      ).join(""),
    };
    console.log(`Open http://localhost:${port} in Chrome. Ctrl+C stops the server.`);
    const abort = new AbortController();
    const stop = (): void => abort.abort();
    Deno.addSignalListener("SIGINT", stop);
    try {
      await Deno.serve(
        { hostname: "127.0.0.1", port, signal: abort.signal },
        createHandler(bundle, config),
      ).finished;
    } finally {
      Deno.removeSignalListener("SIGINT", stop);
    }
  } finally {
    await Deno.remove(build, { recursive: true });
  }
};
if (import.meta.main) await main();
