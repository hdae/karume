import { createFileRoute } from "@tanstack/react-router";

import { AnimaPage } from "@/models/anima/page";

export const Route = createFileRoute("/anima")({ component: AnimaPage });
