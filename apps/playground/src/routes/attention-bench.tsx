import { createFileRoute } from "@tanstack/react-router";

import { AttentionBenchPage } from "@/models/attention-bench/page";

export const Route = createFileRoute("/attention-bench")({ component: AttentionBenchPage });
