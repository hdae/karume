/**
 * anima の DiT 常駐（ADR 0112）を Chrome（WebGPU）で確かめるページのローカルサーバ。
 *
 * 連続 generate・常駐の on / off・ダミー確保で VRAM を埋めた後の退避（`evicted` / `headroom` と
 * `evicted` / `out-of-memory`）を、Chrome の device で観測する。速度計測の比較ページではない。
 *
 * 配信の形は `tools/llm-speed/browser/server.ts` と同じ（localhost 限定・COOP / COEP / CORP・
 * 起動時に `deno bundle --platform browser`・モデルは根の中に閉じて Range 対応で配る）。
 * 区間配信と根の閉じ込めはそちらの実装をそのまま使う（同じ規則を二重に持たない）。
 */
import { containedPath, fileResponse } from "../../llm-speed/browser/server.ts";

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
  /** 配布形ディレクトリの名前（パスは出さない — 手元の構成をページへ漏らさない）。 */
  readonly source: string;
};

/**
 * 手元の checkout の版と未コミットの変更の有無（ページの `/config.json` と Deno の双子 CLI
 * `../profile.ts` の JSON が同じ取り方で載せる）。
 */
export const readCheckout = async (): Promise<{ revision: string; dirty: boolean }> => {
  const git = await new Deno.Command("git", { args: ["rev-parse", "HEAD"] }).output();
  if (!git.success) throw Error("Cannot identify checkout revision");
  const status = await new Deno.Command("git", { args: ["status", "--porcelain"] }).output();
  if (!status.success) throw Error("Cannot inspect checkout changes");
  return {
    revision: new TextDecoder().decode(git.stdout).trim(),
    dirty: status.stdout.length > 0,
  };
};

export const createHandler = (
  sourceRoot: string,
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
      if (url.pathname.startsWith("/models/anima/")) {
        const path = url.pathname.slice("/models/anima/".length);
        return await fileResponse(req, await containedPath(sourceRoot, path));
      }
      return new Response(null, { status: 404, headers: h });
    } catch (error) {
      if (error instanceof Deno.errors.NotFound) {
        return new Response("Asset not found", { status: 404, headers: headers() });
      }
      console.error(error);
      return new Response("Asset request failed; see server log", {
        status: 400,
        headers: headers(),
      });
    }
  };
};

const main = async (): Promise<void> => {
  if (Deno.args.includes("--help")) {
    console.log("deno task bench:anima-browser [--port 8788] [--source models/karume-anima]");
    return;
  }
  const args = new Map<string, string>();
  for (let i = 0; i < Deno.args.length; i += 2) {
    const key = Deno.args[i], value = Deno.args[i + 1];
    if (
      !["--port", "--source"].includes(key) || value === undefined || value.startsWith("--") ||
      args.has(key)
    ) throw Error(`Invalid option ${key}`);
    args.set(key, value);
  }
  const port = Number(args.get("--port") ?? 8788);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw Error("Invalid port");
  const sourceRoot = args.get("--source") ?? "models/karume-anima";
  const realSource = await Deno.realPath(sourceRoot);
  if (!(await Deno.stat(`${realSource}/karume.json`)).isFile) {
    throw Error(`${sourceRoot} has no karume.json`);
  }
  const build = await Deno.makeTempDir({ prefix: "karume-anima-residency-" });
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
      source: realSource.slice(realSource.lastIndexOf("/") + 1),
    };
    console.log(`Open http://localhost:${port} in Chrome. Ctrl+C stops the server.`);
    const abort = new AbortController();
    const stop = (): void => abort.abort();
    Deno.addSignalListener("SIGINT", stop);
    try {
      await Deno.serve(
        { hostname: "127.0.0.1", port, signal: abort.signal },
        createHandler(realSource, bundle, config),
      ).finished;
    } finally {
      Deno.removeSignalListener("SIGINT", stop);
    }
  } finally {
    await Deno.remove(build, { recursive: true });
  }
};
if (import.meta.main) await main();
