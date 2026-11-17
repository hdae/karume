/**
 * The browser's `openInput`: a file under the repository's `inputs/` read by ranged requests to the
 * dev server's `/inputs/` route (`@karume/local-distributions/plugin`), so a file larger than one
 * ArrayBuffer is never read whole. The size comes from a HEAD request. Every URL is resolved against
 * `base`, the page's `document.baseURI`.
 *
 * SPEC: docs/spec/workbench.md#input-files
 */

import type { OpenInput } from "../entry.ts";
import { inputSegments } from "../inputs.ts";

export const browserInputs =
  (base: string): OpenInput =>
  async (path) => {
    const segments = inputSegments(path);
    const url = new URL(`inputs/${segments.map(encodeURIComponent).join("/")}`, base);
    const head = await fetch(url, { method: "HEAD" });
    const size = Number(head.headers.get("Content-Length"));
    if (!head.ok || !Number.isSafeInteger(size)) {
      throw new Error(
        `input file inputs/${path}: the dev server answers ${head.status} (a production build serves no input file)`,
      );
    }
    return {
      file: {
        size,
        read: async (offset, length) => {
          if (length === 0) return new Uint8Array(new ArrayBuffer(0));
          const response = await fetch(url, {
            headers: { Range: `bytes=${offset}-${offset + length - 1}` },
          });
          const bytes = new Uint8Array(await response.arrayBuffer());
          if (response.status !== 206 || bytes.byteLength !== length) {
            throw new Error(
              `input file inputs/${path}: the range [${offset}, ${offset + length}) gave ${response.status} with ${bytes.byteLength} bytes`,
            );
          }
          return bytes;
        },
      },
      facts: { path: `inputs/${segments.join("/")}`, size, read: "direct" },
    };
  };
