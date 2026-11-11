import { createFileRoute } from "@tanstack/react-router";

import { SessionAbPage } from "@/models/session-ab/page";

export const Route = createFileRoute("/session-ab")({ component: SessionAbPage });
