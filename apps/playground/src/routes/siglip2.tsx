import { createFileRoute } from "@tanstack/react-router";

import { Siglip2Page } from "@/models/siglip2/page";

export const Route = createFileRoute("/siglip2")({ component: Siglip2Page });
