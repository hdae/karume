import { createFileRoute } from "@tanstack/react-router";

import { Gemma4BenchPage } from "@/models/gemma4-bench/page";

export const Route = createFileRoute("/gemma4-bench")({ component: Gemma4BenchPage });
