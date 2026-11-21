import { createFileRoute } from "@tanstack/react-router";

import { EmbeddingGemmaPage } from "@/models/embeddinggemma/page";

export const Route = createFileRoute("/embeddinggemma")({ component: EmbeddingGemmaPage });
