import { createFileRoute } from "@tanstack/react-router";

import { MossImagePage } from "@/models/moss-image/page";

export const Route = createFileRoute("/moss-image")({ component: MossImagePage });
