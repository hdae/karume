import { versions } from "./config.ts";

type Roots = { normal: string; qat: string; onnx?: string; vendor?: string };
const headers = (): Headers =>
  new Headers({
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
  });
const contentType = (name: string): string =>
  name.endsWith(".wasm")
    ? "application/wasm"
    : /\.(m?js)$/.test(name)
    ? "text/javascript"
    : name.endsWith(".html")
    ? "text/html; charset=utf-8"
    : name.endsWith(".json")
    ? "application/json"
    : "application/octet-stream";

/** 指定根の外と symlink の越境を拒否する。モデルの既存ファイルは読み取り専用。 */
export const containedPath = async (root: string, path: string): Promise<string> => {
  const segments = decodeURIComponent(path).split("/");
  if (segments.some((s) => !s || s === "." || s === ".." || s.includes("\\") || s.includes("\0"))) {
    throw Error("Invalid asset path");
  }
  const realRoot = await Deno.realPath(root);
  const resolved = await Deno.realPath(`${realRoot}/${segments.join("/")}`);
  if (!resolved.startsWith(`${realRoot}/`)) throw Error("Asset outside model directory");
  return resolved;
};

export const fileResponse = async (req: Request, path: string): Promise<Response> => {
  const file = await Deno.open(path, { read: true });
  let closed = false;
  const close = (): void => {
    if (!closed) {
      closed = true;
      file.close();
    }
  };
  try {
    const stat = await file.stat();
    if (!stat.isFile) throw Error("Not a regular file");
    const h = headers();
    h.set("Content-Type", contentType(path));
    h.set("Accept-Ranges", "bytes");
    const range = req.headers.get("Range");
    let start = 0, end = stat.size - 1;
    if (range !== null) {
      const match = /^bytes=(\d+)-(\d+)$/.exec(range);
      if (
        !match || !Number.isSafeInteger(Number(match[1])) ||
        !Number.isSafeInteger(Number(match[2])) || Number(match[1]) > Number(match[2]) ||
        Number(match[2]) >= stat.size
      ) {
        close();
        h.set("Content-Range", `bytes */${stat.size}`);
        return new Response(null, { status: 416, headers: h });
      }
      start = Number(match[1]);
      end = Number(match[2]);
      h.set("Content-Range", `bytes ${start}-${end}/${stat.size}`);
    }
    let remaining = end - start + 1;
    h.set("Content-Length", String(remaining));
    if (req.method === "HEAD") {
      close();
      return new Response(null, { status: range ? 206 : 200, headers: h });
    }
    await file.seek(start, Deno.SeekMode.Start);
    const body = new ReadableStream<Uint8Array<ArrayBuffer>>({
      async pull(controller): Promise<void> {
        try {
          if (remaining === 0) {
            close();
            controller.close();
            return;
          }
          const buffer = new Uint8Array(Math.min(remaining, 1024 * 1024));
          const n = await file.read(buffer);
          if (n === null) throw Error("Asset ended before Content-Length");
          remaining -= n;
          controller.enqueue(buffer.subarray(0, n));
        } catch (error) {
          close();
          controller.error(error);
        }
      },
      cancel(): void {
        close();
      },
    });
    return new Response(body, { status: range ? 206 : 200, headers: h });
  } catch (error) {
    close();
    throw error;
  }
};

const vendorFiles = (): Record<string, string> => ({
  "transformers.js":
    `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${versions.transformers}/dist/transformers.web.js`,
  ...Object.fromEntries(
    [
      "ort.webgpu.min.mjs",
      "ort-wasm-simd-threaded.asyncify.mjs",
      "ort-wasm-simd-threaded.asyncify.wasm",
    ].map((
      name,
    ) => [
      name,
      `https://cdn.jsdelivr.net/npm/onnxruntime-web@${versions.onnxruntime}/dist/${name}`,
    ]),
  ),
});

export const createHandler = (
  roots: Roots,
  bundle: Uint8Array<ArrayBuffer>,
  revision: string,
  bundleSha256?: string,
  dirty?: boolean,
): (req: Request) => Promise<Response> => {
  const staticRoot = decodeURIComponent(new URL(".", import.meta.url).pathname);
  const vendors = vendorFiles();
  return async (req) => {
    const url = new URL(req.url);
    if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) {
      return new Response("Localhost only", { status: 403 });
    }
    if (req.method !== "GET" && req.method !== "HEAD") return new Response(null, { status: 405 });
    try {
      const h = headers();
      if (url.pathname === "/config.json") {
        return Response.json({
          localOnnx: roots.onnx !== undefined,
          versions,
          revision,
          bundleSha256,
          dirty,
        }, {
          headers: h,
        });
      }
      if (url.pathname === "/runner.js") {
        h.set("Content-Type", "text/javascript");
        return new Response(bundle, { headers: h });
      }
      if (url.pathname === "/frame.html") {
        const html = await Deno.readTextFile(`${staticRoot}/frame.html`);
        h.set("Content-Type", "text/html; charset=utf-8");
        return new Response(html.replaceAll("{{onnxruntime}}", versions.onnxruntime), {
          headers: h,
        });
      }
      const staticFile = new Map([["/", "index.html"], [
        "/cases.json",
        "cases.json",
      ]]).get(url.pathname);
      if (staticFile) return await fileResponse(req, `${staticRoot}/${staticFile}`);
      const model = /^\/models\/(normal|qat)\/(.+)$/.exec(url.pathname);
      if (model) {
        const root = model[1] === "normal" ? roots.normal : roots.qat;
        return await fileResponse(req, await containedPath(root, model[2]));
      }
      if (url.pathname.startsWith("/onnx/") && roots.onnx) {
        return await fileResponse(req, await containedPath(roots.onnx, url.pathname.slice(6)));
      }
      if (url.pathname.startsWith("/vendor/")) {
        const name = url.pathname.slice(8);
        if (!Object.hasOwn(vendors, name)) return new Response(null, { status: 404, headers: h });
        if (roots.vendor) return await fileResponse(req, await containedPath(roots.vendor, name));
        const upstream = await fetch(vendors[name]);
        if (!upstream.ok) {
          await upstream.body?.cancel();
          throw Error(`Dependency HTTP ${upstream.status}: ${name}`);
        }
        h.set("Content-Type", contentType(name));
        h.set("Cache-Control", "public, max-age=86400");
        if (upstream.headers.has("content-length") && !upstream.headers.has("content-encoding")) {
          h.set("Content-Length", upstream.headers.get("content-length")!);
        }
        return new Response(upstream.body, { headers: h });
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
    console.log(
      "deno task bench:llm-browser [--port 8787] [--normal models/karume-gemma4] [--qat models/karume-gemma4-qat] [--onnx <normal/ and qat/ directory>] [--vendor <dependency directory>]",
    );
    return;
  }
  const args = new Map<string, string>();
  for (let i = 0; i < Deno.args.length; i += 2) {
    const key = Deno.args[i], value = Deno.args[i + 1];
    if (
      !["--port", "--normal", "--qat", "--onnx", "--vendor"].includes(key) || value === undefined ||
      value.startsWith("--") || args.has(key)
    ) throw Error(`Invalid option ${key}`);
    args.set(key, value);
  }
  const port = Number(args.get("--port") ?? 8787);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw Error("Invalid port");
  const roots: Roots = {
    normal: args.get("--normal") ?? "models/karume-gemma4",
    qat: args.get("--qat") ?? "models/karume-gemma4-qat",
    onnx: args.get("--onnx"),
    vendor: args.get("--vendor"),
  };
  const build = await Deno.makeTempDir({ prefix: "karume-browser-speed-" });
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
    if (!(await command.output()).success) throw Error("Browser bundle failed");
    const bundle = await Deno.readFile(output);
    const git = await new Deno.Command("git", { args: ["rev-parse", "HEAD"] }).output();
    if (!git.success) throw Error("Cannot identify checkout revision");
    const revision = new TextDecoder().decode(git.stdout).trim();
    const status = await new Deno.Command("git", { args: ["status", "--porcelain"] }).output();
    if (!status.success) throw Error("Cannot inspect checkout changes");
    const dirty = status.stdout.length > 0;
    const bundleSha256 = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bundle)),
      (v) => v.toString(16).padStart(2, "0"),
    ).join("");
    console.log(`Open http://localhost:${port} in Chrome. Ctrl+C stops the server.`);
    const abort = new AbortController();
    const stop = (): void => abort.abort();
    Deno.addSignalListener("SIGINT", stop);
    try {
      await Deno.serve(
        { hostname: "127.0.0.1", port, signal: abort.signal },
        createHandler(roots, bundle, revision, bundleSha256, dirty),
      ).finished;
    } finally {
      Deno.removeSignalListener("SIGINT", stop);
    }
  } finally {
    await Deno.remove(build, { recursive: true });
  }
};
if (import.meta.main) await main();
