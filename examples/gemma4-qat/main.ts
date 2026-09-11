/** gemma4-qat の対話 CLI。 */
import { runGemmaCli } from "../shared/gemma-main.ts";
import { runMain } from "../shared/run-main.ts";

if (import.meta.main) await runMain(() => runGemmaCli("gemma4-qat"));
