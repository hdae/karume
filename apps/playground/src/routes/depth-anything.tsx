import { createFileRoute } from "@tanstack/react-router";

import { DepthAnythingPage } from "@/models/depth-anything/page";

export const Route = createFileRoute("/depth-anything")({ component: DepthAnythingPage });
