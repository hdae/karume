import { createFileRoute } from "@tanstack/react-router";

import { Gemma4Page } from "@/models/gemma4/page";

export const Route = createFileRoute("/gemma4")({ component: Gemma4Page });
