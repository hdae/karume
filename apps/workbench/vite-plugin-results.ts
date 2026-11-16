/**
 * The workbench's server side.
 *
 * - Dev server only: `POST /workbench/results` writes a record under `outputs/workbench/`
 *   ({@link handleRecordPost}) and `GET /workbench/checkout` answers the checkout at request time.
 *   Neither exists in preview or in a production build, which offers a download only.
 * - Dev and preview servers: every response carries COOP / COEP, so the page is cross-origin
 *   isolated (full-precision timers).
 * - Production build: the checkout is baked in as the `__WORKBENCH_CHECKOUT__` constant, read when
 *   the config loads.
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { fileURLToPath } from "node:url";
import type { Connect, Plugin } from "vite-plus";

import { readCheckout } from "./src/checkout.ts";
import { handleRecordPost, MAX_RECORD_BYTES, type RecordPostResponse } from "./src/results.ts";

/** Headers on every dev and preview response (cross-origin isolation of the page). */
export const ISOLATION_HEADERS: Readonly<Record<string, string>> = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

const REPOSITORY_ROOT = fileURLToPath(new URL("../../", import.meta.url));

const isolation: Connect.NextHandleFunction = (_request, response, next) => {
  for (const [name, value] of Object.entries(ISOLATION_HEADERS)) response.setHeader(name, value);
  next();
};

const headersOf = (request: IncomingMessage): Record<string, string | undefined> => {
  const headers: Record<string, string | undefined> = {};
  for (const [name, value] of Object.entries(request.headers)) {
    headers[name] = Array.isArray(value) ? value.join(", ") : value;
  }
  return headers;
};

const send = (response: ServerResponse, answer: RecordPostResponse): void => {
  response.statusCode = answer.status;
  for (const [name, value] of Object.entries(answer.headers)) response.setHeader(name, value);
  response.end(answer.body);
};

const answerWorkbench = async (
  request: IncomingMessage,
  response: ServerResponse,
): Promise<boolean> => {
  const path = (request.url ?? "/").split("?", 1)[0];
  if (path === "/workbench/results") {
    send(
      response,
      await handleRecordPost(
        {
          method: request.method ?? "GET",
          scheme:
            "encrypted" in request.socket && request.socket.encrypted === true ? "https" : "http",
          headers: headersOf(request),
          body: request,
        },
        { repositoryRoot: REPOSITORY_ROOT, now: () => new Date(), maxBytes: MAX_RECORD_BYTES },
      ),
    );
    return true;
  }
  if (path === "/workbench/checkout") {
    if (request.method !== "GET") {
      send(response, {
        status: 405,
        headers: { Allow: "GET", "Content-Type": "text/plain; charset=utf-8" },
        body: "Only GET\n",
      });
    } else {
      send(response, {
        status: 200,
        headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
        body: `${JSON.stringify(await readCheckout(REPOSITORY_ROOT))}\n`,
      });
    }
    return true;
  }
  return false;
};

const workbenchEndpoints: Connect.NextHandleFunction = (request, response, next) => {
  answerWorkbench(request, response).then(
    (answered) => {
      if (!answered) next();
    },
    (error: unknown) => next(error),
  );
};

export const resultsPlugin = (): Plugin => ({
  name: "karume-workbench-results",
  async config(_config, { command }) {
    if (command !== "build") return undefined;
    return {
      define: { __WORKBENCH_CHECKOUT__: JSON.stringify(await readCheckout(REPOSITORY_ROOT)) },
    };
  },
  configureServer(server) {
    server.middlewares.use(isolation);
    server.middlewares.use(workbenchEndpoints);
  },
  configurePreviewServer(server) {
    server.middlewares.use(isolation);
  },
});
