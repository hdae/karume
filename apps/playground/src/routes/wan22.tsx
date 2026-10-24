import { createFileRoute } from "@tanstack/react-router";

import { Wan22Page } from "@/models/wan22/page";

export const Route = createFileRoute("/wan22")({ component: Wan22Page });
