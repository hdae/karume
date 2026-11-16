/**
 * What a production build takes in place of `local-entries.ts`: no entry that reads a local
 * distribution (a production build serves none).
 */

import type { Entry } from "./entry.ts";

export const LOCAL_MODULES: Readonly<Record<string, Entry>> = {};
