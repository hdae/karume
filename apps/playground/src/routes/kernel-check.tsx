import { createFileRoute } from "@tanstack/react-router";

import { KernelCheckPage } from "@/models/kernel-check/page";

export const Route = createFileRoute("/kernel-check")({ component: KernelCheckPage });
