/**
 * The dev server's `POST /workbench/results`, which writes the record a page sends through the
 * shared record writer (`writeRecord` of `@karume/sweep-app-support/node`: `outputs/workbench/<host>-
 * <adapter name>/<entry id>_<UTC stamp>.json` under the repository root, created exclusively). Node
 * only.
 *
 * Nothing the client sends becomes a path segment except through the writer's two rules: the entry
 * id must be kebab-case, and the adapter directory keeps only `[a-z0-9-]`. The stamp is the
 * server's clock.
 *
 * SPEC: docs/spec/workbench.md#where-records-go
 * SPEC: docs/spec/workbench.md#the-dev-server
 */

import { writeRecord } from "@karume/sweep-app-support/node";
import { ENTRY_ID, RECORD_FORMAT } from "@karume/sweep-app-support/record";
import type { RecordTarget } from "@karume/sweep-app-support/record-path";

/** The largest record the dev server takes. */
export const MAX_RECORD_BYTES = 64 * 1024 * 1024;

export type RecordPostRequest = {
  readonly method: string;
  readonly scheme: "http" | "https";
  /** Lower-case names, as Node gives them. */
  readonly headers: Readonly<Record<string, string | undefined>>;
  /** Read only as far as {@link RecordPostContext.maxBytes} allows. */
  readonly body: AsyncIterable<Uint8Array>;
};

export type RecordPostContext = {
  readonly repositoryRoot: string;
  readonly now: () => Date;
  readonly maxBytes: number;
};

export type RecordPostResponse = {
  readonly status: number;
  readonly headers: Readonly<Record<string, string>>;
  readonly body: string;
};

const textResponse = (
  status: number,
  text: string,
  headers: Readonly<Record<string, string>> = {},
): RecordPostResponse => ({
  status,
  headers: { "Content-Type": "text/plain; charset=utf-8", ...headers },
  body: `${text}\n`,
});

/** The body, or `undefined` once it passes `maxBytes` (nothing past the cap is kept or read). */
const readCapped = async (
  body: AsyncIterable<Uint8Array>,
  maxBytes: number,
): Promise<Uint8Array<ArrayBuffer> | undefined> => {
  const chunks: Uint8Array[] = [];
  let length = 0;
  // The iterator is not closed on the way out: closing a request stream destroys the socket before
  // the 413 reaches the client. The response says `Connection: close` instead.
  const iterator = body[Symbol.asyncIterator]();
  for (;;) {
    const step = await iterator.next();
    if (step.done === true) break;
    length += step.value.byteLength;
    if (length > maxBytes) return undefined;
    chunks.push(step.value);
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** The parts of a posted record the path needs, or why it is not a workbench record. */
const postedTarget = (bytes: Uint8Array): RecordTarget | string => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    return "the body is not UTF-8 JSON";
  }
  if (!isRecord(parsed) || parsed["format"] !== RECORD_FORMAT) {
    return `format is not ${JSON.stringify(RECORD_FORMAT)}`;
  }
  const entry = parsed["entry"];
  const entryId = isRecord(entry) ? entry["id"] : undefined;
  if (typeof entryId !== "string" || !ENTRY_ID.test(entryId)) {
    return "entry.id is not a kebab-case id";
  }
  const environment = parsed["environment"];
  const host = isRecord(environment) ? environment["host"] : undefined;
  if (host !== "browser" && host !== "node") return "environment.host is not browser or node";
  const adapter = isRecord(environment) ? environment["adapter"] : undefined;
  if (!isRecord(adapter)) return "environment.adapter is not an object";
  return { entryId, host, adapter };
};

/**
 * `POST /workbench/results`: writes the posted record, unchanged, through {@link writeRecord} and
 * answers 201 with `{ path }`. The `Origin` header MUST equal the origin the request was sent to,
 * so another site open in the same browser cannot write files (a cross-site POST carries its own
 * origin; Vite's host check, which runs first, rejects a rebound DNS name).
 */
export const handleRecordPost = async (
  request: RecordPostRequest,
  context: RecordPostContext,
): Promise<RecordPostResponse> => {
  if (request.method !== "POST") return textResponse(405, "Only POST", { Allow: "POST" });
  if (!/^\s*application\/json\s*(;|$)/i.test(request.headers["content-type"] ?? "")) {
    return textResponse(415, "Content-Type must be application/json");
  }
  const host = request.headers["host"];
  const origin = request.headers["origin"];
  if (host === undefined || origin !== `${request.scheme}://${host}`) {
    return textResponse(403, "The Origin header must be this server's origin");
  }
  const bytes = await readCapped(request.body, context.maxBytes);
  if (bytes === undefined) {
    return textResponse(413, `The record is larger than ${context.maxBytes} bytes`, {
      Connection: "close",
    });
  }
  const target = postedTarget(bytes);
  if (typeof target === "string") return textResponse(422, target);
  const path = await writeRecord(context.repositoryRoot, target, bytes, context.now());
  return {
    status: 201,
    headers: { "Content-Type": "application/json" },
    body: `${JSON.stringify({ path })}\n`,
  };
};
