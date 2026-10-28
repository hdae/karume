import { createFileRoute } from "@tanstack/react-router";

import { Sbv2Page } from "@/models/sbv2/page";

export const Route = createFileRoute("/sbv2")({ component: Sbv2Page });
