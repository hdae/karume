import { createFileRoute } from "@tanstack/react-router";

import { ExpertTransferPage } from "@/models/expert-transfer/page";

export const Route = createFileRoute("/expert-transfer")({ component: ExpertTransferPage });
