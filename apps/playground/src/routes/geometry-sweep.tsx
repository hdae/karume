import { createFileRoute } from "@tanstack/react-router";

import { GeometrySweepPage } from "@/models/geometry-sweep/page";

export const Route = createFileRoute("/geometry-sweep")({ component: GeometrySweepPage });
