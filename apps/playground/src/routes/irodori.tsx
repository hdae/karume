import { createFileRoute } from "@tanstack/react-router";

import { IrodoriPage } from "@/models/irodori/page";

export const Route = createFileRoute("/irodori")({ component: IrodoriPage });
