/** qwen3 のローカル変換済みモデルを使う単発チャット CLI。 */
import { runLlmCli } from "../shared/llm-main.ts";
import { runMain } from "../shared/run-main.ts";

if (import.meta.main) await runMain(() => runLlmCli("qwen3"));
