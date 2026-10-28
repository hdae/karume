import { createFileRoute } from "@tanstack/react-router";

import { BirefnetPage } from "@/models/birefnet/page";

export const Route = createFileRoute("/birefnet")({ component: BirefnetPage });
