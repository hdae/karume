/**
 * The modules of the entries that read a local distribution (`src/entries/local/`), for the
 * registry. A production build resolves this module to `no-local-entries.ts` (`vite.config.ts`), so
 * those entries and everything only they import are outside its module graph — not merely unlisted.
 */

import type { Entry } from "./entry.ts";

export const LOCAL_MODULES: Readonly<Record<string, Entry>> = import.meta.glob<Entry>(
  "./entries/local/*.ts",
  { eager: true, import: "default" },
);
